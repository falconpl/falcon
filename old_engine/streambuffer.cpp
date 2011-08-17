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

#include <falcon/memory.h>
#include <falcon/streambuffer.h>
#include <falcon/fassert.h>

#include <string.h>
#include <cstring>

namespace Falcon {

StreamBuffer::StreamBuffer( Stream *underlying, bool bOwn, uint32 bufSize ):
   Stream( underlying->type() ),
   m_bufSize( bufSize ),
   m_changed( false ),
   m_bufPos(0),
   m_bufLen(0),
   m_filePos(0),
   m_bReseek( false ),
   m_stream( underlying ),
   m_streamOwner( bOwn )
{
   m_buffer = (byte *) memAlloc( m_bufSize );
}

StreamBuffer::StreamBuffer( const StreamBuffer &other ):
   Stream( other.m_streamType ),
   m_bufSize( other.m_bufSize ),
   m_changed( other.m_changed ),
   m_bufPos( other.m_bufPos ),
   m_bufLen( other.m_bufLen ),
   m_filePos( other.m_filePos ),
   m_bReseek( other.m_bReseek ),
   m_streamOwner( other.m_streamOwner )
{
   if( m_streamOwner )
      m_stream = dyncast<Stream*>(other.m_stream->clone());
   else
      m_stream = other.m_stream;

   m_buffer = (byte *) memAlloc( m_bufSize );
}


StreamBuffer::~StreamBuffer()
{
   flush();

   if( m_streamOwner )
      delete m_stream;

   memFree( m_buffer );
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
      m_filePos += m_bufLen;
   }

   m_bufPos = 0;

   int32 readIn = m_stream->read( m_buffer, m_bufSize );
   if ( readIn < 0 )
   {
      m_bufLen = 0;
      return false;
   }

   // changed was set to false from flush
   m_bufLen = readIn;

   return true;
}

int32 StreamBuffer::read( void *b, int32 size )
{
   // minimal sanity check
   if ( size <= 0 )
      return 0;

   if( ! m_stream->good() || ! m_stream->open() )
      return -1;

   byte *buf = (byte*) b;

   int32 avail = m_bufLen - m_bufPos;

   // have we something to store in the buffer?
   if ( avail > 0 )
   {
      // enough to fill the buffer?
      if ( avail >= size )
      {
         memcpy( buf, m_buffer + m_bufPos, size );
         m_bufPos += size;
         return size;
      }
      else {
         // in the meanwhile, put the data in.
         memcpy( buf, m_buffer + m_bufPos, avail );
         m_bufPos = m_bufLen;  // declare we have consumed everything.
         // return a partial read in case of underlying networks
         if ( m_stream->type() == t_network )
            return avail;
      }
   }

   // if we're here, we need to refill the buffer, or eventually to read everything from the stream
   // the amount of data we still have to put in the buffer is size - avail.

   int32 toBeRead = size - avail;

   // would be a new buffer enough to store the data?
   if ( toBeRead <= m_bufSize )
   {
      if ( ! refill() )
      {
         // if the refill operation failed, return what we have read.
         return m_stream->bad() ? -1 : avail;
      }

      // if the refill operation succed, it is still possible that it has read less than toBeRead.
      if ( m_bufLen < toBeRead )
         toBeRead = m_bufLen;

      memcpy( buf + avail, m_buffer, toBeRead );
      m_bufPos = toBeRead;  // declare we have consumed the data.
      return toBeRead + avail;
   }
   else
   {
      // it's of no use to refill the buffer now. Just read the required size and update the file pointer.
      m_filePos += m_bufLen;  // a buffer is gone.

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


int32 StreamBuffer::write( const void *b, int32 size )
{
   // minimal sanity check
   if ( size <= 0 )
      return 0;

   if( m_stream->status() != t_open )
      return -1;

   const byte *buf = (byte*) b;

   // first; is there any space left in the buffer for write?
   int32 avail = m_bufSize - m_bufPos;
   if( avail > 0 )
   {
      m_changed = true;

      // good; if we have enough space, advance and go away.
      if ( size <= avail )
      {
         memcpy( m_buffer + m_bufPos, buf, size );
         m_bufPos += size;
         if ( m_bufLen < m_bufPos )
         {
            m_bufLen = m_bufPos;
         }

         return size;
      }
      else {
         // nay, we can write only part of the stuff.
         memcpy( m_buffer + m_bufPos, buf, avail );
         m_bufPos = m_bufLen = m_bufSize;
      }
   }

   // we have still to write part or all the data.

   // now, if the rest of the data can be stored in the next buffer,
   // refill and write. Otherwise, just flush and try a single write out.
   int32 toBeWritten = size - avail;
   if( toBeWritten <= m_bufSize )
   {
      flush();
      m_changed = true; // ensure we declare this buffer changed anyhow

      memcpy( m_buffer, buf + avail, toBeWritten );
      m_bufPos = m_bufLen = toBeWritten;

      return avail + toBeWritten;
   }
   else
   {
      flush(); // but don't reload now

      toBeWritten = m_stream->write( buf + avail, toBeWritten );
      if( toBeWritten < 0 )
      {
         return avail;
      }
      m_filePos += toBeWritten;
      return avail + toBeWritten;
   }
}


bool StreamBuffer::close()
{
   flush();
   return m_stream->close();
}

int64 StreamBuffer::tell()
{
   return m_filePos + m_bufPos;
}

bool StreamBuffer::truncate( int64 pos )
{
   if ( pos == -1 )
   {
      // shorten the buffer to the current position
      m_bufLen = m_bufPos;
      // And truncate
      if(  m_stream->truncate( m_filePos + m_bufPos ) )
      {
         flush();
         return true;
      }

      return false;
   }
   else
   {
      int64 curpos = m_filePos + m_bufPos;
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

int32 StreamBuffer::readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData )
{
   if ( m_bufPos < m_bufLen )
      return m_bufLen - m_bufPos;

   return m_stream->readAvailable( msecs_timeout, sysData );
}

int32 StreamBuffer::writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData )
{
   if ( m_bufPos < m_bufSize )
      return m_bufSize - m_bufPos;

   return m_stream->writeAvailable( msecs_timeout, sysData );
}

int64 StreamBuffer::seek( int64 pos, e_whence whence )
{
   // TODO: optimize and avoid re-buffering if we're still in the buffer.
   if( whence == ew_cur )
   {
      pos = m_filePos + m_bufPos + pos;
      whence = ew_begin;
   }

   flush();
   m_bufLen = m_bufPos = 0;

   m_filePos = m_stream->seek( pos, whence );
   m_bReseek = false;
   return m_filePos;
}

bool StreamBuffer::flush()
{
   if ( ! m_changed || m_bufLen == 0 )
      return true;

   int32 written = m_stream->write( m_buffer, m_bufLen );
   int32 count = written;
   while( written > 0 && count < m_bufLen )
   {
      written = m_stream->write( m_buffer + count, m_bufLen - count );
      count += written;
   }

   if( written < 0 )
      return false;

   m_filePos += m_bufPos;
   m_bReseek = m_bReseek || m_bufPos != m_bufLen;
   m_changed = false;
   m_bufPos = m_bufLen = 0;

   m_stream->flush();

   return true;
}

bool StreamBuffer::get( uint32 &chr )
{
   if ( popBuffer(chr) )
      return true;

   if ( m_bufPos == m_bufLen )
   {
      if ( ! refill() )
         return false;

      if( m_bufPos == m_bufLen ) // eof?
         return false;
   }

   chr = m_buffer[m_bufPos++ ];
   return true;
}

bool StreamBuffer::put( uint32 chr )
{
   if ( m_bufPos == m_bufSize )
   {
      if ( ! flush() )
         return false;

      m_buffer[ 0 ] = chr;
      m_bufPos = m_bufLen = 1;
      m_changed = true;
      return true;
   }

   m_buffer[ m_bufPos++ ] = chr;
   m_changed = true;
   if ( m_bufLen < m_bufPos )
      m_bufLen = m_bufPos;
   return true;
}

bool StreamBuffer::resizeBuffer( uint32 size )
{
   fassert( size > 0 );
   if ( ! flush() )
      return false;

   memFree( m_buffer );
   m_buffer = (byte*) memAlloc( size );
   m_bufSize = size;
   return true;
}

}


/* end of streambuffer.cpp */
