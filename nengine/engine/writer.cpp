/*
   FALCON - The Falcon Programming Language.
   FILE: writer.cpp

   Base abstract class for input stream readers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 12:56:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/writer.h>
#include <falcon/sys.h>
#include <falcon/stream.h>
#include <falcon/ioerror.h>

#include <stdio.h>
#include <string.h>


namespace Falcon {

Writer::Writer():
   m_buffer(0),
   m_bufPos(0),
   m_bufSize(0),
   m_bOwnStream(false),
   m_stream(0)
{}

Writer::Writer( Stream* stream, bool bOwn ):
   m_bOwnStream(bOwn),
   m_stream(stream)
{
   m_bufSize = Sys::_getPageSize();
   m_bufPos = 0;
   m_buffer = new byte[m_bufSize];
}


Writer::~Writer()
{
   delete[] m_buffer;
   if ( m_bOwnStream )
   {
      delete m_stream;
   }
}

void Writer::delegate( Writer& target )
{
   if( target.m_bOwnStream )
   {
      delete target.m_stream;
   }

   delete[] target.m_buffer;

   target.m_stream = this->m_stream;
   target.m_bOwnStream = this->m_bOwnStream;

   target.m_buffer = this->m_buffer;
   target.m_bufPos = this->m_bufPos;
   target.m_bufSize = this->m_bufSize;

   this->m_buffer = 0;
   this->m_stream = 0;
   this->m_bOwnStream = false;
}

void Writer::setBufferSize( length_t bs )
{
   fassert( m_stream != 0 );
   fassert( bs > 0 );

   m_bufSize = bs;

   byte* nBuf = new byte[m_bufSize];
   if ( m_bufPos > 0 )
   {
      memmove( nBuf, m_buffer, m_bufPos );
   }

   delete[] m_buffer;
   m_buffer = nBuf;
}


bool Writer::flush()
{
   size_t nDone = 0;
   while ( nDone < m_bufPos)
   {
      size_t nSize = m_stream->write( m_buffer + nDone, m_bufPos - nDone );

      if ( nSize == (size_t) -1 )
      {
         // if the stream wanted to throw, we wouldn't be here.
         return false;
      }

      nDone += nSize;
   }

   m_bufPos = 0;
   return true;
}

void Writer::ensure( size_t size )
{
   if ( size + m_bufPos > m_bufSize )
   {
      setBufferSize( size + m_bufPos );
   }
}

bool Writer::write( byte* data, size_t size )
{
   if( size + m_bufPos <= m_bufSize )
   {
      memcpy( m_buffer + m_bufPos, data, size );
      m_bufPos += size;
   }
   else
   {
      size_t nDone = m_bufSize - m_bufPos;
      memcpy( m_buffer + m_bufPos, data, nDone );
      if( ! flush() )
      {
         return false;
      }

      // write a multiple of the buffer size
      size_t toWrite = ((size-nDone) / m_bufSize)*m_bufSize + nDone;

      while( nDone < toWrite )
      {
         size_t nWritten = m_stream->write( data + nDone, toWrite - nDone );
         if( nWritten < (size_t)-1 )
         {
            return false;
         }
         nDone += nWritten;
      }

      fassert( size - nDone > 0 );
      fassert( size - nDone < m_bufSize );

      m_bufPos = size - nDone;
      memcpy( m_buffer, data + nDone, m_bufPos );
   }

   return true;
}


 void Writer::changeStream( Stream* s, bool bOwn, bool bDiscard )
 {
    if( bDiscard )
    {
       m_bufPos = 0;
    }
    else
    {
       flush();
    }

    if ( m_bOwnStream )
    {
       delete s;
    }

    m_bOwnStream = bOwn;
    m_stream = s;
 }

}

/* end of writer.cpp */
