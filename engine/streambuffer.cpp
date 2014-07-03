/*
   FALCON - The Falcon Programming Language
   FILE: streambuffer.cpp

   Buffer wrapper for stream operations.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 02 Feb 2009 16:45:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Buffer wrapper for stream operations.
*/

#if defined BSD || defined __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <falcon/streambuffer.h>
#include <falcon/fassert.h>
#include <falcon/stdhandlers.h>

#include <string.h>
#include <cstring>

namespace Falcon {

StreamBuffer::StreamBuffer( Stream *underlying, uint32 bufSize ):
   m_bufSize( bufSize ),
   m_changed( false ),
   m_rBufPos(0),
   m_rBufLen(0),
   m_wBufPos(0),
   m_wBufLen(0),
   m_filePos(0),
   m_bReseek( false ),
   m_stream( underlying )
{
   m_stream->incref();
   m_rbuffer = (byte *) malloc( m_bufSize );
   if( underlying->hasPipeSemantic() )
   {
      m_wbuffer = (byte *) malloc( m_bufSize );
   }
   else {
      m_wbuffer = m_rbuffer;
   }
}

StreamBuffer::StreamBuffer( const StreamBuffer &other ):
   Stream( other ),
   m_bufSize( other.m_bufSize ),
   m_changed( other.m_changed ),
   m_rBufPos( other.m_rBufPos ),
   m_rBufLen( other.m_rBufLen ),
   m_wBufPos( other.m_wBufPos ),
   m_wBufLen( other.m_wBufLen ),
   m_filePos( other.m_filePos ),
   m_bReseek( other.m_bReseek )
{
   m_stream = other.m_stream;
   m_stream->incref();
   m_rbuffer = (byte *) malloc( m_bufSize );
   memcpy( m_rbuffer, other.m_rbuffer, other.m_rBufLen );

   if( m_stream->hasPipeSemantic() )
   {
      m_wbuffer = (byte *) malloc( m_bufSize );
      memcpy( m_wbuffer, other.m_wbuffer, other.m_wBufLen );
   }
   else {
      m_wbuffer = m_rbuffer;
   }
}


StreamBuffer::~StreamBuffer()
{
   flush();
   bool freeWrite = m_stream->hasPipeSemantic();
   m_stream->decref();
   free( m_rbuffer );
   if( freeWrite )
   {
      free(m_wbuffer);
   }
}


StreamBuffer *StreamBuffer::clone() const
{
   return new StreamBuffer( *this );
}


bool StreamBuffer::refill()
{
   if ( ! flush() )
      return false;

   if( m_bReseek )
   {
      m_stream->seekBegin( m_filePos );
      m_bReseek = false;
   }
   else {
      m_filePos += m_rBufLen;
   }

   m_rBufPos = 0;
   if( ! m_stream->hasPipeSemantic() )
   {
      m_wBufPos = m_rBufPos;
   }

   int32 readIn = m_stream->read( m_rbuffer, m_bufSize );
   if ( readIn < 0 )
   {
      m_rBufLen = 0;
      if( ! m_stream->hasPipeSemantic() )
      {
         m_wBufLen = m_rBufLen;
      }
      return false;
   }

   // changed was set to false from flush
   m_rBufLen = readIn;
   if( ! m_stream->hasPipeSemantic() )
   {
      m_wBufLen = m_rBufLen;
   }

   return true;
}

size_t StreamBuffer::read( void *b, size_t size )
{
   // minimal sanity check
   if ( size == 0 )
      return 0;

   if( ! m_stream->good() || ! m_stream->open() )
      return -1;

   byte *buf = (byte*) b;

   size_t avail = m_rBufLen - m_rBufPos;

   // have we something to store in the buffer?
   if ( avail > 0 )
   {
      // enough to fill the buffer?
      if ( avail >= size )
      {
         memcpy( buf, m_rbuffer + m_rBufPos, size );
         m_rBufPos += size;
         return size;
      }
      else {
         // in the meanwhile, put the data in.
         memcpy( buf, m_rbuffer + m_rBufPos, avail );
         m_rBufPos = m_rBufLen;  // declare we have consumed everything.
         // return a partial read in case of underlying networks
         if ( m_stream->hasPipeSemantic() )
         {
            return avail;
         }

      }
   }

   // if we're here, we need to refill the buffer, or eventually to read everything from the stream
   // the amount of data we still have to put in the buffer is size - avail.

   size_t toBeRead = size - avail;

   // would be a new buffer enough to store the data?
   if ( toBeRead <= m_bufSize )
   {
      if ( ! refill() )
      {
         // if the refill operation failed, return what we have read.
         return m_stream->bad() ? -1 : avail;
      }

      // if the refill operation succeed, it is still possible that it has read less than toBeRead.
      if ( m_rBufLen < toBeRead )
      {
         toBeRead = m_rBufLen;
      }

      memcpy( buf + avail, m_rbuffer, toBeRead );
      m_rBufPos = toBeRead;  // declare we have consumed the data.
      if( ! m_stream->hasPipeSemantic() )
      {
         m_wBufPos = m_rBufPos;
         m_wBufLen = m_rBufLen;
      }
      return toBeRead + avail;
   }
   else
   {
      // it's of no use to refill the buffer now. Just read the required size and update the file pointer.
      m_filePos += m_rBufLen;  // a buffer is gone.

      int32 readin = m_stream->read( buf + avail, size - avail );
      if( readin > 0 )
      {
          m_filePos += readin;
          return readin + avail;
      }
      else {
         // we have an error, but return the read data.
         return avail;
      }
   }
}


size_t StreamBuffer::write( const void *b, size_t size )
{
   // minimal sanity check
   if ( size == 0 )
      return 0;

   if( m_stream->status() != t_open )
      return -1;

   const byte *buf = (byte*) b;

   // first; is there any space left in the buffer for write?
   size_t avail = m_bufSize - m_wBufPos;
   if( avail > 0 )
   {
      m_changed = true;

      // good; if we have enough space, advance and go away.
      if ( size <= avail )
      {
         memcpy( m_wbuffer + m_wBufPos, buf, size );
         m_wBufPos += size;
         if ( m_wBufLen < m_wBufPos )
         {
            m_wBufLen = m_wBufPos;
         }

         if( ! m_stream->hasPipeSemantic() )
         {
            m_rBufPos = m_wBufPos;
            m_rBufLen = m_wBufLen;
         }

         return size;
      }
      else {
         // nay, we can write only part of the stuff.
         memcpy( m_wbuffer + m_wBufPos, buf, avail );
         m_wBufPos = m_wBufLen = m_bufSize;
      }
   }

   // we have still to write part or all the data.

   // now, if the rest of the data can be stored in the next buffer,
   // refill and write. Otherwise, just flush and try a single write out.
   size_t toBeWritten = size - avail;
   if( toBeWritten <= m_bufSize )
   {
      flush();
      m_changed = true; // ensure we declare this buffer changed anyhow

      memcpy( m_wbuffer, buf + avail, toBeWritten );
      m_wBufPos = m_wBufLen = toBeWritten;
      if( ! m_stream->hasPipeSemantic() )
      {
         m_rBufPos = m_rBufLen = toBeWritten;
      }

      return avail + toBeWritten;
   }
   else
   {
      flush(); // but don't reload now

      int written = m_stream->write( buf + avail, toBeWritten );
      if( written < 0 )
      {
         return avail;
      }
      m_filePos += written;
      return avail + written;
   }
}


bool StreamBuffer::close()
{
   flush();
   return m_stream->close();
}

int64 StreamBuffer::tell()
{
   return m_filePos + m_rBufPos;
}

bool StreamBuffer::truncate( int64 pos )
{
   if ( pos == -1 )
   {
      // shorten the buffer to the current position
      m_rBufLen = m_rBufPos;
      // And truncate
      if(  m_stream->truncate( m_filePos + m_rBufPos ) )
      {
         flush();
         return true;
      }

      return false;
   }
   else
   {
      int64 curpos = m_filePos + m_rBufPos;
      flush();

      // and trunk at the desired position
      if ( pos < curpos )
      {
         m_filePos = pos;
         curpos = pos;
      }

      m_stream->seekBegin( curpos );
      return m_stream->truncate( pos );
   }
}

int64 StreamBuffer::seek( int64 pos, e_whence whence )
{
   // TODO: optimize and avoid re-buffering if we're still in the buffer.
   if( whence == ew_cur )
   {
      pos = m_filePos + m_rBufPos + pos;
      whence = ew_begin;
   }

   flush();
   m_rBufLen = m_rBufPos = 0;

   m_filePos = m_stream->seek( pos, whence );
   m_bReseek = false;
   return m_filePos;
}

bool StreamBuffer::flush()
{
   if ( ! m_changed || m_wBufLen == 0 )
      return true;

   int32 written = m_stream->write( m_wbuffer, m_wBufLen );
   uint32 count = 0;
   while( written > 0 && count < m_wBufLen )
   {
      count += (uint32) written;
      written = m_stream->write( m_wbuffer + count, m_wBufLen - count );
   }

   if( written < 0 )
      return false;

   m_filePos += m_wBufPos;
   m_bReseek = m_bReseek || m_wBufPos != m_wBufLen;
   m_changed = false;
   m_wBufPos = m_wBufLen = 0;
   if( ! m_stream->hasPipeSemantic() )
   {
      m_rBufPos = m_rBufLen = 0;
   }

   m_stream->flush();

   return true;
}


bool StreamBuffer::get( uint32 &chr )
{
   if ( m_rBufPos == m_rBufLen )
   {
      if ( ! refill() )
         return false;

      if( m_rBufPos == m_rBufLen ) // eof?
         return false;
   }

   chr = m_rbuffer[m_rBufPos++ ];
   return true;
}


bool StreamBuffer::put( uint32 chr )
{
   if ( m_wBufPos == m_bufSize )
   {
      if ( ! flush() )
         return false;

      m_wbuffer[ 0 ] = chr;
      m_wBufPos = m_wBufLen = 1;
      m_changed = true;
      return true;
   }

   m_wbuffer[ m_rBufPos++ ] = chr;
   m_changed = true;
   if ( m_wBufLen < m_wBufPos )
      m_wBufLen = m_wBufPos;
   return true;
}

bool StreamBuffer::resizeBuffer( uint32 size )
{
   fassert( size > 0 );
   
   if(size != m_bufSize )
   {
      if ( ! flush() )
      {
         return false;
      }

      free( m_rbuffer );
      m_rbuffer = (byte*) malloc( size );
      m_bufSize = size;
      m_rBufLen = m_rBufPos = 0;

      if( m_stream->hasPipeSemantic() )
      {
         free(m_wbuffer);
         m_wbuffer = (byte*) malloc( size );
      }
      else {
         m_wbuffer = m_rbuffer;
      }
   }
   
   return true;
}


const Multiplex::Factory* StreamBuffer::multiplexFactory() const
{
   return m_stream->multiplexFactory();
}

}


/* end of streambuffer.cpp */
