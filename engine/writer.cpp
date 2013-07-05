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
#include <falcon/stderrors.h>

#include <stdio.h>
#include <string.h>


namespace Falcon {

Writer::Writer():
   m_buffer(0),
   m_bufPos(0),
   m_bufSize(0),
   m_gcMark(0),
   m_stream(0)
{}

Writer::Writer( Stream* stream ):
   m_gcMark(0),
   m_stream(stream)
{
   if( stream != 0 )
   {
      stream->incref();
   }

   m_bufSize = Sys::_getPageSize();
   m_bufPos = 0;
   m_buffer = new byte[m_bufSize];
}


Writer::Writer( const Writer& other ):
   m_gcMark(0)
{
   if( other.m_stream != 0 )
   {
      other.m_stream->incref();
   }

   m_stream = other.m_stream;
   m_bufSize = other.m_bufSize;
   m_bufPos = other.m_bufPos;
   m_buffer = new byte[m_bufSize];

   memcpy( m_buffer, other.m_buffer, other.m_bufSize );
}


Writer::~Writer()
{
   flush();
   delete[] m_buffer;
   if ( m_stream != 0 )
   {
      m_stream->decref();
   }
}

void Writer::delegate( Writer& target )
{
   if( target.m_stream )
   {
      target.m_stream->decref();
   }

   delete[] target.m_buffer;

   target.m_stream = this->m_stream;

   target.m_buffer = this->m_buffer;
   target.m_bufPos = this->m_bufPos;
   target.m_bufSize = this->m_bufSize;

   this->m_bufPos = 0;
   this->m_bufSize = 0;
   this->m_buffer = 0;
   this->m_stream = 0;

}

void Writer::setBufferSize( length_t bs )
{
   m_mtx.lock();
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
   m_mtx.unlock();
}


bool Writer::flush()
{
   m_mtx.lock();
   size_t nDone = 0;
   while ( nDone < m_bufPos)
   {
      size_t nSize = m_stream->write( m_buffer + nDone, m_bufPos - nDone );

      if ( nSize == (size_t) -1 )
      {
         // if the stream wanted to throw, we wouldn't be here.
         m_mtx.unlock();
         return false;
      }

      nDone += nSize;
   }

   m_bufPos = 0;
   m_mtx.unlock();
   return true;
}

void Writer::ensure( size_t size )
{
   m_mtx.lock();
   if ( size + m_bufPos > m_bufSize )
   {
      length_t bp = m_bufPos;
      m_mtx.unlock();

      setBufferSize( size + bp );
   }
   else {
      m_mtx.unlock();
   }
}

bool Writer::writeRaw( byte* data, size_t size )
{
   m_mtx.lock();
   if( size + m_bufPos <= m_bufSize )
   {
      memcpy( m_buffer + m_bufPos, data, size );
      m_bufPos += size;
   }
   else
   {
      size_t nDone = m_bufSize - m_bufPos;
      memcpy( m_buffer + m_bufPos, data, nDone );
      m_bufPos += nDone; // flush uses current bufPos to save.

      m_mtx.unlock();
      if( ! flush() )
      {
         return false;
      }

      m_mtx.lock();
      // write a multiple of the buffer size
      size_t toWrite = ((size-nDone) / m_bufSize)*m_bufSize + nDone;

      while( nDone < toWrite )
      {
         size_t nWritten = m_stream->write( data + nDone, toWrite - nDone );
         if( nWritten < (size_t)-1 )
         {
            m_mtx.unlock();
            return false;
         }
         nDone += nWritten;
      }

      fassert( size - nDone > 0 );
      fassert( size - nDone < m_bufSize );

      m_bufPos = size - nDone;
      memcpy( m_buffer, data + nDone, m_bufPos );
   }
   m_mtx.unlock();

   return true;
}


 void Writer::changeStream( Stream* s, bool bDiscard )
 {
    if( bDiscard )

    if( ! bDiscard )
    {
       flush();
    }

    if( s != m_stream )
    {
       if( s != 0 )
       {
          s->incref();
       }

       m_mtx.lock();
       if ( m_stream != 0 )
       {
          m_stream->decref();
       }

       if( bDiscard )
       {
          m_bufPos = 0;
       }

       m_stream = s;
       m_mtx.unlock();
    }


 }

}

/* end of writer.cpp */
