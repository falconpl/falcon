/*
   FALCON - The Falcon Programming Language.
   FILE: request.cpp

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 12:29:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/request.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/utils.h>
#include <falcon/wopi/wopi_opts.h>
#include <falcon/wopi/errors.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/uploaded.h>
#include <falcon/wopi/classuploaded.h>
#include <falcon/log.h>

#include <falcon/uri.h>
#include <falcon/error.h>
#include <falcon/engine.h>
#include <falcon/textwriter.h>
#include <falcon/sys.h>

// for memcpy
#include <string.h>
#include <stdio.h>
#include <ctype.h>

namespace Falcon {
namespace WOPI {


//==================================================================
// Request

Request::Request( ModuleWopi* host ):
   m_content_length( -1 ),
   m_bytes_received( 0 ),
   m_request_time(0),
   m_startedAt(0.0),

   // protected
   m_gets( new ItemDict ),
   m_posts( new ItemDict ),
   m_cookies( new ItemDict ),
   m_headers( new ItemDict )
{
   m_module = host;
   m_MainPart.setOwner( this );

   FALCON_GC_HANDLE( m_gets );
   FALCON_GC_HANDLE( m_posts );
   FALCON_GC_HANDLE( m_cookies );
   FALCON_GC_HANDLE( m_headers );

}

Request::~Request()
{
}


void Request::parse( Stream* input )
{
   try
   {
      parseHeader( input );
      if( m_content_length > 0 )
      {
         parseBody( input );
      }
   }
   catch ( ... )
   {
      Engine::collector()->enable(true);
      throw;
   }
}


void Request::parseHeader( Stream* input )
{
   m_MainPart.parseHeader( input );

   // get the content type and encoding
   PartHandler::HeaderMap::const_iterator ci = m_MainPart.headers().find( "Content-Type" );
   if( ci != m_MainPart.headers().end() )
   {
      m_content_type = ci->second.rawValue();

      // default the content-length to 0 in case of non multipart data.
      if( m_content_type.find( "multipart/") != 0 )
         m_content_length = 0;

   }

   ci = m_MainPart.headers().find( "Content-Encoding" );
   if( ci != m_MainPart.headers().end() )
   {
      m_content_encoding = ci->second.rawValue();
   }

   ci = m_MainPart.headers().find( "Content-Length" );
   if( ci != m_MainPart.headers().end() )
   {
      ci->second.rawValue().parseInt( m_content_length );
      m_MainPart.setBodySize( m_content_length );
   }

   // parse the cookies.
   ci = m_MainPart.headers().find( "Cookie" );
   if( ci != m_MainPart.headers().end() )
   {
      HeaderValue::ParamMap::const_iterator pi = ci->second.parameters().begin();
      while( pi != ci->second.parameters().end() )
      {
         m_cookies->insert(
                        FALCON_GC_HANDLE(&(new String(pi->first))->bufferize()),
                        FALCON_GC_HANDLE(&(new String(pi->second))->bufferize())
                   );
         ++pi;
      }
   }

   ci = m_MainPart.headers().begin();
   while( ci != m_MainPart.headers().end() )
   {
      m_headers->insert(
              FALCON_GC_HANDLE(&(new String(ci->first))->bufferize()),
              FALCON_GC_HANDLE(&(new String(ci->second.rawValue()))->bufferize())
         );
      ++ci;
   }
}


void Request::parseBody( Stream* input )
{
   Log* LOG = Engine::instance()->log();
   m_bytes_received = 0;

   // read the configuration relevant keys
   String error;
   int64 maxUpload = 0;
   int64 maxMemUpload = 0;
   if( module() != 0 )
   {
      module()->wopi()->getConfigValue(OPT_MaxUploadSize, maxUpload, error);
      module()->wopi()->getConfigValue(OPT_MaxMemoryUploadSize, maxMemUpload, error);
   }

   LOG->log(Log::fac_engine_io, Log::lvl_detail,
            String("Receiving upload with Content-Length: ").N(m_content_length)
               .A("/").N(maxUpload).A( " mem:").N(maxMemUpload) );

   if( maxUpload > 0 && m_content_length > maxUpload *1024 )
   {
      throw FALCON_SIGN_XERROR(WopiError, FALCON_ERROR_WOPI_UPLOAD_TOO_BIG, .extra("") );
   }
   // prepare the POST data receive area
   m_MainPart.startMemoryUpload();

   // Inform the part if it can use memory uploads for their subparts.
   if ( m_content_length != -1 &&
         (maxMemUpload > 0 && m_content_length < maxMemUpload*1024) )
   {
      // This tell the children of the main part NOT TO create a temporary file
      // when they receive a file upload (the default).
      // Standard form fields are still received in memory.
      m_MainPart.uploadsInMemory( true );
   }
   // For prudence,

   bool bDummy = false;
   m_MainPart.parseBody( input, bDummy );

   // shouldn't be necessary as we started the upload in memory mode.
   m_MainPart.closeUpload();

   // it's a singlepart or multipart?
   PartHandler* child = m_MainPart.child();
   if( child == 0 )
   {
      // parse the post fields
      String post_data;
      m_MainPart.getMemoryData( post_data );
      Falcon::WOPI::Utils::parseQuery( post_data, *m_posts );
   }
   else
   {
      while( child != 0 )
      {
         addUploaded( child );
         child = child->next();
      }
   }

   m_bytes_received = m_content_length;
}


bool Request::getField( const String& fname, String& value ) const
{
   Item temp;
   if( ! getField( fname, temp ) || ! temp.isString() )
      return false;

   value = *temp.asString();
   return true;
}


bool Request::getField( const String& fname, Item& value ) const
{
   Item* res;
   // try in gets.
   if( (res = m_gets->find( fname )) == 0 )
   {
      if( (res = m_posts->find(fname) ) == 0 )
      {
         if( (res = m_cookies->find(fname) ) == 0 )
         {
            return false;
         }
      }
   }

   value = *res;
   return true;
}


void Request::fwdGet( String& fwd, bool all ) const
{
   forward( *m_gets, *m_posts, fwd, all );
}


void Request::fwdPost( String& fwd, bool all ) const
{
   Utils::dictAsInputFields( fwd, *m_posts );
   if( all )
   {
      Utils::dictAsInputFields( fwd, *m_gets );
   }
}

void Request::forward( const ItemDict& main, const ItemDict& aux, String& fwd, bool all ) const
{
   fwd.size(0);
   Utils::fieldsToUriQuery( main, fwd );
   if( all )
   {
      Utils::fieldsToUriQuery( aux, fwd );
   }
}


bool Request::setURI( const String& uri )
{
   if( m_uri.parse( uri ) )
   {
      m_sUri = uri;
      if ( m_uri.query().size() != 0 )
      {
         // we can't trust URI query field, as RFC for URI locator query variables
         // doesn't include special naming conventions as dictionaries and arrays
         // passed in that.

         uint32 pos = uri.find('?');
         WOPI::Utils::parseQuery(uri.subString(pos+1), *m_gets);
      }

      m_location = m_uri.path().encode();
      return true;
   }

   return false;
}


//================================================================
// Falcon interface.
//


void Request::addUploaded( PartHandler* ph, const String& prefix )
{
   String key = prefix.size() == 0 ? ph->name(): prefix + "." + ph->name();

   if( ph->isFile() )
   {
      // an empty file field?
      if( ph->filename().size() == 0 )
      {
         // puts a nil item.
         Falcon::WOPI::Utils::addQueryVariable( key, Item(), *m_posts );
      }
      else
      {
         // configure the part
         Uploaded* upld = new Uploaded(ph->filename(), ph->contentType(), ph->uploadedSize());
         // no? -- store the data or the temporary file name.
         if ( ph->isMemory() )
         {
            String* data = new String;
            ph->getMemoryData( *data );
            upld->data(data);
         }
         else
         {
            upld->storage(ph->tempFile());
         }

         // It may take some time before we can reach the vm,
         // so it's better to be sure we're not marked.
         Falcon::WOPI::Utils::addQueryVariable( key, FALCON_GC_STORE(m_module->uploadedClass(), upld), *m_posts );
      }
   }
   else
   {
      // We have just to add this field.
      if( ph->isMemory() )
      {
         String temp;
         String* value = new String;
         ph->getMemoryData( temp );
         temp.c_ize();
         value->fromUTF8( (char *) temp.getRawStorage() );

         // It may take some time before we can reach the vm,
         // so it's better to be sure we're not marked.
         Falcon::WOPI::Utils::addQueryVariable( key, FALCON_GC_HANDLE(value), *m_posts );
      }
      // else, don't know what to do
   }

   PartHandler* child = ph->child();
   while( child != 0 )
   {
      addUploaded( child, key );
      child = child->next();
   }
}


bool Request::processMultiPartBody()
{
   PartHandler* child = m_MainPart.child();
   if( child == 0 )
   {
      return false;
   }

   while( child != 0 )
   {
      addUploaded( child );
      child = child->next();
   }

   return true;
}


//================================================================
// CGI-oriented
//
void Request::parseEnviron()
{
   // First; suck all the environment variables that we need.
   Falcon::Sys::_enumerateEnvironment( &handleEnvStr, this );

   // a bit of post-processing
   if ( parsedUri().auth().port() == "443" || parsedUri().auth().port() == "https" )
   {
      parsedUri().scheme("https");
   }
   else
   {
      parsedUri().scheme("http");
   }
}

void Request::handleEnvStr( const Falcon::String& key, const Falcon::String& value, void *data )
{
   // First; suck all the environment variables that we need.
   Request* self = static_cast<Request*>(data);

   // Is this an header transformed in an env-var?
   if( key.startsWith("HTTP_") )
   {
      self->addHeaderFromEnv( key, value );
   }
   else if( key == "AUTH_TYPE" )
   {
      self->m_ap_auth_type = value;
   }
   else if( key == "CONTENT_TYPE" )
   {
      self->m_content_type = value;
      self->m_MainPart.addHeader( "Content-Type", value );
   }
   else if( key == "CONTENT_LENGTH" )
   {
      Falcon::int64 tgt;
      value.parseInt(tgt);
      self->m_content_length = (int) tgt;
   }
   else if( key == "DOCUMENT_ROOT" )
   {
      // ....
   }
   else if( key == "GATEWAY_INTERFACE" )
   {
      // ....
   }
   else if( key == "PATH_INFO" )
   {
      self->m_path_info = value;
   }
   else if ( key == "QUERY_STRING" )
   {
      // it's part of the REQUEST_URI
   }
   else if( key == "REMOTE_ADDR" )
   {
      self->m_remote_ip = value;
   }
   else if( key == "REMOTE_PORT" )
   {
      //... not implemented
   }
   else if( key == "REMOTE_USER" )
   {
      self->m_user = value;
   }
   else if( key == "REQUEST_METHOD" )
   {
      self->m_method = value;
   }
   else if( key == "REQUEST_URI" )
   {
      self->setURI( value );
   }
   else if( key == "SCRIPT_FILENAME" )
   {
      self->m_filename = value;
   }
   else if( key == "SCRIPT_NAME" )
   {
      self->parsedUri().path() = value;
   }
   else if( key == "SERVER_ADDR" )
   {
      // ....
   }
   else if( key == "SERVER_ADMIN" )
   {
      // ....
   }
   else if( key == "SERVER_NAME" )
   {
      self->parsedUri().auth().host( value );
   }
   else if( key == "SERVER_PORT" )
   {
      self->parsedUri().auth().port( value );
   }
   else if( key == "SERVER_PROTOCOL" )
   {
      self->m_protocol = value;
   }
   else if( key == "SERVER_SIGNATURE" )
   {
      // ....
   }
   else if( key == "SERVER_SOFTWARE" )
   {
      // ....
   }
}

void Request::addHeaderFromEnv( const Falcon::String& key, const Falcon::String& value )
{
   // discard "http_"
   Falcon::String sKey( key, 5 );

   // ... and transform the rest in "Content-Type" format
   bool bUpper = true;
   Falcon::uint32 len = sKey.length();

   for( Falcon::uint32 i = 0; i < len; ++i )
   {
      if( sKey[i] == '_' )
      {
         sKey.setCharAt(i, '-');
         bUpper = true;
      }
      else if ( bUpper )
      {
         sKey.setCharAt(i, toupper( sKey[i] ) );
         bUpper = false;
      }
      else
      {
         sKey.setCharAt(i, tolower( sKey[i] ) );
      }
   }

   // Ok, we can now add the thing to the dict
   headers()->insert( FALCON_GC_HANDLE(new String( sKey )), FALCON_GC_HANDLE(new String( value )) );

   if ( key == "HTTP_COOKIE" )
   {
      Falcon::uint32 pos = 0;
      Falcon::uint32 pos1 = value.find(";");
      while( true )
      {
         Falcon::WOPI::Utils::parseQueryEntry( value.subString(pos,pos1), *cookies() );

         if( pos1 == Falcon::String::npos )
            break;

         pos = pos1+1;
         pos1 = value.find(";", pos );
      }

   }
   else if ( key == "HTTP_CONTENT_TYPE" )
   {
      m_content_type = value;
   }
   else if ( key == "HTTP_CONTENT_ENCODING" )
   {
      m_content_encoding = value;
   }
}



//================================================================
// Override
//

void Request::gcMark( uint32 mark )
{
   if( mark != m_mark )
   {
      m_mark = mark;
      m_gets->gcMark( mark );
      m_posts->gcMark( mark );
      m_cookies->gcMark( mark );
      m_headers->gcMark( mark );
   }
}

}
}

/* end of request.cpp */
