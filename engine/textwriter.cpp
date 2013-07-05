/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.cpp

   Text-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/textwriter.h>
#include <falcon/stream.h>
#include <falcon/engine.h>
#include <falcon/transcoder.h>
#include <falcon/stderrors.h>
#include <string.h>

#include <list>


namespace Falcon {

TextWriter::TextWriter( Stream* stream ):
   Writer( stream ),
   m_bWasCR(false),
   m_bCRLF(false),
   m_bLineFlush(false),
   m_twBufSize(0),
   m_twBuffer(0),
   m_mark(0)
{
   m_encoder = Engine::instance()->getTranscoder("C");
   m_chrStr.reserve(2);
}


TextWriter::TextWriter( Stream* stream, Transcoder* decoder ):
   Writer( stream ),
   m_encoder( decoder ),
   m_bWasCR(false),
   m_bCRLF(false),
   m_bLineFlush(false),
   m_twBufSize(0),
   m_twBuffer(0),
   m_mark(0)
{
   m_chrStr.reserve(2);
}


TextWriter::~TextWriter()
{
   flush();
   delete[] m_twBuffer;
}

void TextWriter::setEncoding( Transcoder* decoder )
{
   m_encoder = decoder;
}

void TextWriter::setSysCRLF()
{
#ifdef FALCON_SYSTEM_UNIX
   m_bCRLF = false;
#else
   m_bCRLF = true;
#endif
}

bool TextWriter::write( const String& str, length_t start, length_t count )
{
   fassert( start <= str.length() )
   if ( count == String::npos )
   {
      count = str.length() - start;
   }
   
   // fast path?
   if( ! (m_bCRLF || m_bLineFlush ) )
   {
      return rawWrite( str, start, count );      
   }

   length_t end = start + count;
   length_t pos1 = start;

   while(true)
   {
      length_t posNext = str.findFirstOf( "\r\n", pos1 );
      // ok also when not found.
      if( posNext >= end )
      {
         // we're done -- no more \n (so we can't flush).
         return rawWrite( str, pos1, end - pos1 );
      }
      
      if( m_bCRLF && (posNext == 0 || str.getCharAt(posNext-1) != '\r') )
      {
         // write separately the \r\n sequence
         if( posNext > 0 && ! rawWrite( str, pos1, posNext-pos1 ) )
         {
            return false;
         }

         m_mtx.lock();
         ensure(m_encoder->encodingSize(2));
         m_bufPos += m_encoder->encode("\r\n", currentBuffer(), m_bufSize - m_bufPos );
         m_mtx.unlock();
      }
      else
      {
         // just write everything, \n included.
         if( ! rawWrite( str, pos1, posNext-pos1 +1 ) )
         {
            return false;
         }
      }

      if ( m_bLineFlush )
      {
         if( ! flush() ) return false;
      }

      pos1 = posNext+1; // we know we're in the string, or we would have been exited.
   }

   return true;
}


bool TextWriter::rawWrite( const String& str, length_t start, length_t count )
{
   if( count == String::npos )
   {
      count = str.length() - start;
   }
   
   m_mtx.lock();
   length_t encSize = m_encoder->encodingSize(count);
   if( encSize < m_bufSize - m_bufPos )
   {
      m_bufPos += m_encoder->encode( str, currentBuffer(), m_bufSize - m_bufPos,
         '?', start, count );
      m_mtx.unlock();
      return true;
   }
   else
   {
      // use the internal buffer to rely on write,
      // -- and so, allow flushing at multiple of page size.
      if( encSize > m_twBufSize )
      {
         delete[] m_twBuffer;
         m_twBuffer = new byte[encSize];
         m_twBufSize = encSize;
      }

      encSize = m_encoder->encode( str, m_twBuffer, m_twBufSize, '?', start, count );
      bool res = Writer::writeRaw( m_twBuffer, encSize );
      m_mtx.unlock();
      return res;
   }

   m_mtx.unlock();
   return true;
}


bool TextWriter::writeLine( const String& str, length_t start, length_t count )
{
   if( ! rawWrite( str, start, count ) ) return false;
   return putChar( '\n' );
}


bool TextWriter::putChar( char_t chr )
{
   byte buf[16];
   m_mtx.lock();
   m_chrStr.size(0);

   if( chr == '\r')
   {
      m_bWasCR = true;
      m_chrStr.append( '\r' );
   }
   else if( chr == '\n' )
   {
      if ( ! m_bWasCR && m_bCRLF )
      {
         m_chrStr.append('\r');
      }
      
      m_chrStr.append('\n');
      m_bWasCR = false;
   }
   else
   {
      m_chrStr.append( chr );
   }

   length_t rsize = m_encoder->encode( m_chrStr, buf, 16 );
   if( ! Writer::writeRaw( buf, rsize ) ) {
      m_mtx.unlock();
      return false;
   }

   if( m_bLineFlush && chr == '\n' )
   {
      bool res = flush();
      m_mtx.unlock();
      return res;
   }
   m_mtx.unlock();
   return true;
}


}

/* end of textreader.cpp */
