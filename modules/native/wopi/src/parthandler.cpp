/*
   FALCON - The Falcon Programming Language.
   FILE: parthandler.cpp

   Web Oriented Programming Interface

   RFC1867 compliant upload handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 26 Feb 2010 20:29:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/parthandler.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/utils.h>
#include <falcon/wopi/errors.h>
#include <falcon/stringstream.h>
#include <falcon/fstream.h>
#include <falcon/error.h>

#include <cstring>

#include <stdio.h>

namespace Falcon {
namespace WOPI {

HeaderValue::HeaderValue()
{
}

HeaderValue::HeaderValue( const String& value )
{
   parseValuePart( value );
}

HeaderValue::HeaderValue( const HeaderValue& other ):
   m_lValues( other.m_lValues ),
   m_mParameters( other.m_mParameters )
{
}

HeaderValue::~HeaderValue()
{
}

bool HeaderValue::parseValue( const String& v )
{
   m_sRawValue = v;

   uint32 pos=0, pos1=0;

   // get the value
   pos = v.find(";");
   if( pos == String::npos )
   {
      // eventually, parse it also as parameters
      // --- but don't complain if it's not in the format V=N
      parseParamPart( v );
      return parseValuePart( v );
   }

   // else

   if( ! parseValuePart( v.subString(0,pos) ) )
      return false;

   // eventually, parse it also as parameters
   // --- but don't complain if it's not in the format V=N
   parseParamPart( v.subString(0,pos) );


   // now parse all the parameters
   ++pos;
   while( (pos1 = v.find(";", pos) ) != String::npos )
   {
      if( ! parseParamPart( v.subString(pos, pos1) ) )
         return false;

      pos = pos1+1;
   }

   // the last parameter
   return parseParamPart( v.subString(pos) );
}

bool HeaderValue::parseValuePart( const String& v )
{
   uint32 pos=0, pos1=0;
   // now parse all the parameters
   while( (pos1 = v.find(",", pos) ) != String::npos )
   {
      m_lValues.push_back( URI::URLDecode( v.subString(pos, pos1) ) );
      m_lValues.back().trim();
      pos = pos1+1;
   }

   m_lValues.push_back( URI::URLDecode( v.subString(pos) ) );
   m_lValues.back().trim();
   return true;
}


bool HeaderValue::parseParamPart( const String& v )
{
   uint32 pos = v.find( "=" );

   if( pos == String::npos )
      return false;

   String sKey = v.subString(0,pos);
   sKey.trim();

   String sValue = v.subString(pos+1);
   sValue.trim();

   if( sValue.startsWith("\"") && sValue.endsWith("\"") )
   {
      sValue = sValue.subString(1,sValue.length()-1);
   }

   // prevent misunderstading effectively empty values
   if(  sValue.size() == 0 )
   {
      m_mParameters[sKey] = "";
      return true;
   }

   // Is URI encoded?
   String sValFinal = URI::URLDecode( sValue );
   if( sValFinal.size() == 0 )
   {
      // if it's not URI encoded, then it must be UTF-8 encoded
      // also, no valid utf-8 encoding can be uri encoded, so...
      String sValDec;
      sValue.c_ize();
      sValDec.fromUTF8( (char*) sValue.getRawStorage() );
      m_mParameters[sKey] = sValDec;

   }
   else
   {
      m_mParameters[sKey] = sValFinal;
   }

   return true;
}

void HeaderValue::setRawValue( const String& v )
{
   m_sRawValue = v;
   m_lValues.push_front( v );
}

//======================================================
//

PartHandler::PartHandler():
   m_pBuffer( 0 ),
   m_bOwnBuffer( false ),

   m_nPartSize(-1),
   // Initially, think we're the main part.
   m_pToBodyLeft(&m_nBodyLeft),
   m_nBodyLeft(1),
   m_nReceived(0),
   m_stream(0),
   m_str_stream(0),
   m_pNextPart( 0 ),
   m_pSubPart( 0 ),
   m_owner( 0 ),
   m_bPartIsFile( false ),
   m_bUseMemoryUpload( false )
{
   m_nBodyLeft <<= 62;
}


PartHandler::PartHandler( const String& sBound ):
   m_pBuffer( 0 ),
   m_bOwnBuffer( false ),

   m_sEnclosingBoundary( sBound ),
   m_nPartSize(-1),
   m_pToBodyLeft(&m_nBodyLeft),
   m_nBodyLeft(1),
   m_nReceived(0),
   m_stream(0),
   m_str_stream(0),
   m_pNextPart( 0 ),
   m_pSubPart( 0 ),
   m_owner( 0 ),
   m_bPartIsFile( false ),
   m_bUseMemoryUpload( false )
{
   m_nBodyLeft <<= 62;
}


PartHandler::~PartHandler()
{
   if( m_stream != 0 )
   {
      m_stream->decref();
   }
   delete m_pNextPart;
   delete m_pSubPart;
   if( m_bOwnBuffer )
      delete m_pBuffer;
}


// Reads a single field in the request
void PartHandler::startMemoryUpload()
{
   m_stream = m_str_stream = new StringStream;
}

void PartHandler::startFileUpload()
{
   m_sTempFile = "";
   Stream* fs;
   try
   {
      fs = m_owner->module()->wopi()->makeTempFile( m_sTempFile, true );
   }
   catch ( Error* e )
   {
      Error* err = FALCON_SIGN_XERROR( WopiError, FALCON_ERROR_WOPI_IO,
               .desc("Generic WOPI I/O error")
               .extra( "Can't create temporary stream " + m_sTempFile )
               );
      err->appendSubError(e);
      e->decref();
      throw err;
   }

   // was this stream first opened in memory?
   if( m_str_stream != 0 )
   {
      try {
         // we must discharge all on disk.
         int wr, written = 0;
         while( (written != (int) m_str_stream->tell()) &&
               ( wr = fs->write( m_str_stream->data()+written, (int) m_str_stream->tell() )) > 0  )
         {
            written += wr;
         }

         m_str_stream->decref();
         m_str_stream = 0;
      }
      catch( ... )
      {
         fs->decref();
         delete m_str_stream;
         m_str_stream = 0;
      }
   }

   m_stream = fs;
}


void PartHandler::closeUpload()
{
   m_nPartSize = m_nReceived = m_stream->tell();

   if( m_str_stream == 0 && m_stream != 0 )
      m_stream->close();
}

bool PartHandler::getMemoryData( String& target )
{
   if( m_str_stream != 0 )
   {
      m_str_stream->closeToString( target );
      delete m_str_stream;
      m_str_stream = 0;
      m_stream = 0;
      return true;
   }

   return false;
}


void PartHandler::startUpload()
{
   if ( m_bUseMemoryUpload || ! m_bPartIsFile )
   {
      startMemoryUpload();
   }
   else
   {
      startFileUpload();
   }
}


void PartHandler::parse( Stream* input, bool& isLast )
{
   // read the header and the body
   parseHeader( input );

   // time to open the upload
   startUpload();

   try {
      parseBody( input, isLast );
      closeUpload();
   }
   catch(...)
   {
      closeUpload();
      throw;
   }
}


void PartHandler::parseBody( Stream* input, bool& isLast )
{
   fassert( m_stream != 0 );

   // If we don't have it, create the buffer to be passed down.
   if( m_pBuffer == 0 )
      initBuffer();

   m_nBodyLeft -= m_pBuffer->m_nBufSize - m_pBuffer->m_nBufPos;

   // is this a mulitpart element?
   if( m_sBoundary.size() != 0 )
   {
      parsePrologue( input );
      parseMultipartBody( input );

      // are we part of a bigger multipart element?
      if ( m_sEnclosingBoundary.size() != 0 )
      {
         scanForBound( "\r\n--"+m_sEnclosingBoundary, input, isLast );
      }
   }
   // are we part of a bigger multipart element?
   else if ( m_sEnclosingBoundary.size() != 0 )
   {
      scanForBound( "\r\n--"+m_sEnclosingBoundary, input, isLast );
   }
   else
   {
      // Save previous data, if any
      m_pBuffer->allFlush(m_stream);

      // just get the (rest of the) body
      while( *m_pToBodyLeft > 0 && (! input->eof()) )
      {
         m_pBuffer->fill( input );
         m_pBuffer->allFlush( m_stream );
      }

      isLast = true;
   }
}


void PartHandler::parsePrologue( Stream* input )
{
   // we must scan to the first body
   bool bIsLast;

   // When processed through web servers, the prologue is removed.
   // shouldn't be the last, or we have no multipart
   scanForBound( "--"+m_sBoundary, input, bIsLast );
   if (bIsLast )
   {
      throw FALCON_SIGN_XERROR( WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE,
               .extra("Can't find the initial boundary") );
   }

}


void PartHandler::parseMultipartBody( Stream* input )
{
   PartHandler* child = new PartHandler( m_sBoundary );
   m_pSubPart = child;
   passSetting( child );

   bool bIsLast;
   child->parse( input, bIsLast );
   while( ! bIsLast )
   {
      child->m_pNextPart = new PartHandler( m_sBoundary );
      child = child->m_pNextPart;
      passSetting( child );
      child->parse( input, bIsLast );
   }
}



void PartHandler::scanForBound( const String& boundary, Stream* input, bool& isLast )
{
   m_pBuffer->fill( input );

   uint32 nBoundPos = m_pBuffer->find( boundary );

   while( nBoundPos == String::npos )
   {
      if( *m_pToBodyLeft == 0 || input->eof() || m_pBuffer->m_nBufPos + boundary.size() + 2 > m_pBuffer->m_nBufSize )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Malformed part (missing boundary)" ));
      }

      m_pBuffer->m_nBufPos = m_pBuffer->m_nBufSize - (boundary.size() + 2);
      m_pBuffer->flush( m_stream );
      m_pBuffer->fill( input );
      nBoundPos = m_pBuffer->find( boundary );
   }

   // We found a match. Is this the last?
   m_pBuffer->m_nBufPos = nBoundPos;

   if( ! m_pBuffer->hasMore( boundary.size() + 2, input, m_stream ) )
   {
      throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Malformed part (missing ending)" ));
   }

   m_pBuffer->flush( m_stream );

   String sRealBound;
   m_pBuffer->grabMore( sRealBound, boundary.size() + 2 );


   if( sRealBound.endsWith( "--" ) )
   {
      // was the last part -- but is it the flux last element?
      if( m_pBuffer->hasMore( boundary.size() + 4, input, m_stream ) )
      {
         m_pBuffer->grabMore( sRealBound, boundary.size() + 4 );

         if( ! sRealBound.endsWith( "\r\n" ) )
         {
            throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Malformed last part (missing trailing CRLF)" ));
         }

         m_pBuffer->m_nBufPos += boundary.size() + 4;
      }
      isLast = true;
   }
   else
   {
      if( ! sRealBound.endsWith( "\r\n" ) )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Malformed part (missing trailing CRLF)" ));
      }

      m_pBuffer->m_nBufPos += boundary.size() + 2;
      isLast = false;
   }

   //remove the boundary -- Necessary?
   m_pBuffer->flush();
}


void PartHandler::parseHeader( Stream* input )
{
   // if we're not given a buffer, create it now
   if ( m_pBuffer == 0 )
      initBuffer();

   m_pBuffer->fill( input );

   uint32 pos;

   while( true )
   {
      pos = m_pBuffer->find( "\r\n" );

      // No more headers? -- get new data
      if( pos == String::npos )
      {
         m_pBuffer->flush();
         // was the flush operation useless -- can't we make new space?
         if( m_pBuffer->full() )
         {
            throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Single header entity too large" ));
         }

         // hit eof in the previous loops?
         if( *m_pToBodyLeft == 0 || input->eof()  )
         {
            throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Unterminated header" ));
         }

         m_pBuffer->fill( input );

         // Re-perform the search
         continue;
      }

      // are we done?
      if( pos == 0 )
      {
         // discard read headers
         m_pBuffer->m_nBufPos += 2;
         break;
      }

      // parse this field
      String sLineBuf;
      m_pBuffer->grabMore( sLineBuf, pos );

      parseHeaderField( sLineBuf );

      // update the buffer pos
      m_pBuffer->m_nBufPos += pos + 2;
   }

   // flush discarding parsed headers.
   m_pBuffer->flush(); // Maybe not necessary here
}


void PartHandler::parseHeaderField( const String& line )
{
   String sKey, sValue;
   if( ! Utils::parseHeaderEntry( line, sKey, sValue ) )
   {
      throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Failed to parse header: " + line ));
   }

   addHeader( sKey, sValue );
}


void PartHandler::addHeader( const String& sKeyu, const String& sValue )
{
   // probably this is an overkill
   String sKey = URI::URLDecode( sKeyu );

   HeaderMap::iterator iadd = m_mHeaders.find( sKey );
   if( iadd == m_mHeaders.end() )
   {
      iadd = m_mHeaders.insert( HeaderMap::value_type( sKey, HeaderValue() ) ).first;
   }

   HeaderValue& hv = iadd->second;

   // we can accept, but not canonicize, invalid parameters
   bool bValid = hv.parseValue( sValue );
   if ( ! bValid )
   {
      hv.setRawValue( sValue );
   }

   // a bit of checks

   if ( sKey.compareIgnoreCase( "Content-Disposition" ) == 0 )
   {
      if( ! bValid )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Failed to parse header value: " + sValue ));
      }

      HeaderValue::ParamMap::const_iterator ni = hv.parameters().find( "name" );
      if ( ni != hv.parameters().end() )
      {
         m_sPartName = ni->second;
      }
      else
      {
         // can't be a well formed content-disposition
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Invalid Content-Disposition" ));
      }

      ni = hv.parameters().find( "filename" );
      if ( ni != hv.parameters().end() )
      {
         // The filename may be empty if the field wasn't sent.
         m_sPartFileName = ni->second;
         m_bPartIsFile = true;
      }
   }
   else if ( sKey.compareIgnoreCase( "Content-Type" ) == 0 )
   {
      if( ! bValid )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Failed to parse header value: " + sValue ));
      }

      if( ! hv.hasValue() )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Invalid Content-Type" ));
      }

      m_sPartContentType = hv.value();
      if( m_sPartContentType.startsWith("multipart/") )
      {
         m_sBoundary = hv.parameters()["boundary"];
         if( m_sBoundary.size() == 0 )
         {
            throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Missing boundary in multipart Content-Type" ));
         }
      }
   }
   else if ( sKey.compareIgnoreCase( "Content-Length" ) == 0 )
   {
      if( ! hv.hasValue() )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Invalid Content-Length" ));
      }

      if (! hv.value().parseInt( m_nPartSize ) )
      {
         throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_REQUEST_PARSE, .extra( "Invalid Content-Length" ));
      }
   }
}

//============================================================
// Buffer management
//

void PartHandler::initBuffer()
{
   if( m_pBuffer != 0 )
      return;

   m_bOwnBuffer = true;
   m_pBuffer = new PartHandlerBuffer( m_pToBodyLeft );
}

void PartHandler::passSetting( PartHandler* child )
{
   child->m_bUseMemoryUpload = m_bUseMemoryUpload;
   child->m_owner = m_owner;
   child->m_pBuffer = m_pBuffer;
   child->m_bOwnBuffer = false;

   // Pass the same pointer of the owner
   child->m_pToBodyLeft = m_pToBodyLeft;
}


PartHandler::PartHandlerBuffer::PartHandlerBuffer( int64* imax ):
      m_nBufPos(0),
      m_nBufSize(0),
      m_nDataLeft( imax )
{
   m_buffer = (byte*) malloc( buffer_size );
}


PartHandler::PartHandlerBuffer::~PartHandlerBuffer()
{
   free( m_buffer );
}

void PartHandler::PartHandlerBuffer::flush( Stream* output )
{
   if( m_nBufPos > 0 )
   {
      if( output != 0 )
         output->write( m_buffer, m_nBufPos );

      // Is there something left in the buffer?
      if( m_nBufPos < m_nBufSize )
      {
         // Roll the buffer.
         // -- are the areas overlapping?
         if( m_nBufSize - m_nBufPos < m_nBufPos )
         {
            // no? -- use a plain memcpy
            memcpy( m_buffer, m_buffer + m_nBufPos, m_nBufSize - m_nBufPos );
         }
         else
         {
            // yes? -- use a slower memmove
            memmove( m_buffer, m_buffer + m_nBufPos, m_nBufSize - m_nBufPos );
         }
      }

      if( m_nBufPos <= m_nBufSize )
         m_nBufSize = m_nBufSize - m_nBufPos;
      else
         m_nBufSize = 0;

      m_nBufPos = 0;
   }
}


uint32 PartHandler::PartHandlerBuffer::find( const String& str )
{
   if( m_nBufPos >= m_nBufSize )
      return String::npos;

   String temp;
   temp.adopt( (char*) m_buffer + m_nBufPos, m_nBufSize - m_nBufPos, 0 );
   return temp.find( str );
}


bool PartHandler::PartHandlerBuffer::fill( Stream* input )
{
   if ( *m_nDataLeft > 0 && (! input->eof()) )
   {
      int32 toRead = buffer_size - m_nBufSize;
      if( toRead > *m_nDataLeft )
      {
         toRead = (int32) *m_nDataLeft;
      }

      int32 size = input->read( m_buffer+m_nBufSize, toRead );
      if( size <= 0 )
         return false;

      m_nBufSize += size;
      *m_nDataLeft -= size;
      return true;
   }

   return false;
}

void PartHandler::PartHandlerBuffer::grab( String& target, int posEnd )
{
   target.adopt( (char*) m_buffer + m_nBufPos, posEnd - m_nBufPos, 0 );
}

void PartHandler::PartHandlerBuffer::grabMore( String& target, int size )
{
   target.adopt( (char*) m_buffer + m_nBufPos, size, 0 );
}

bool PartHandler::PartHandlerBuffer::hasMore( int count, Stream* input, Stream* output )
{
   if( (m_nBufPos + count) < m_nBufSize )
   {
      return true;
   }

   if( (m_nBufPos + count) > buffer_size )
      flush(output);

   fill( input );

   return (m_nBufPos + count) < m_nBufSize;
}

}
}

/* end of parthandler.cpp */
