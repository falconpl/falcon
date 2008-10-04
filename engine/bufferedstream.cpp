/*
   FALCON - The Falcon Programming Language
   FILE: bufferedstream.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/memory.h>
#include <falcon/bufferedstream.h>
#include <cstring>
#include <string.h>

namespace Falcon {

BufferedStream::BufferedStream( uint32 bufSize ):
   m_bufSize( bufSize ),
   m_changed( false ),
   m_bufPos(0),
   m_bufLen(0),
   m_filePos(0)
{
   m_buffer = (byte *) memAlloc( m_bufSize );
   if ( m_buffer == 0 )
   {
      setError( -1 );
      status( t_error );
   }
   else
      status( t_none );
}

BufferedStream::~BufferedStream()
{
   flush();
   delete m_buffer;
}


bool BufferedStream::refill()
{
   if ( ! flush() )
      return false;

   m_filePos = FileStream::tell();
   m_bufPos = 0;
   int32 readIn = FileStream::read( m_buffer, m_bufSize );
   if ( readIn < 0 ) {
      m_bufLen = 0;
      return false;
   }

   m_changed = false;
   m_bufLen = readIn;
   if ( readIn > 0 )
   {
      m_status = t_open;
   }

   return true;
}

int32 BufferedStream::read( void *buf, int32 size )
{
   if ( size <= 0 )
      return 0;

   byte* buffer = (byte*) buf;
   int inbuf = 0;
   // time to read fresh data?
   if ( m_bufLen == m_bufPos )
   {
      if ( size >= m_bufSize )
         return FileStream::read( buffer, size );

      if ( ! refill() )
         return -1;
   }
   else if ( size + m_bufPos > m_bufLen )
   {
      // copy all the left buffer.
      inbuf = m_bufLen - m_bufPos;
      memcpy( buffer, m_buffer + m_bufPos, inbuf );
      m_bufPos = m_bufLen;

      // is the data to ber read still greater than the buffer?
      if( size - inbuf > m_bufSize )
      {
         // don't refill the buffer now.
         int moved = FileStream::read( buffer, size - inbuf );
         if ( moved >= 0 ) {
            m_lastMoved = moved + inbuf;
            if( moved == 0 )
               m_status = m_status | t_eof;
            return m_lastMoved;
         }
         else
            return -1;
      }

      if ( ! refill() )
         return -1;
   }

   m_lastMoved = size;
   int lastIn = size - inbuf;
   if ( lastIn + m_bufPos > m_bufLen )
   {
      lastIn = m_bufLen - m_bufPos;
   }

   if ( lastIn == 1 )
      buffer[inbuf] = m_buffer[ m_bufPos ];
   else
      memcpy( buffer + inbuf , m_buffer + m_bufPos, lastIn );
   m_bufPos += lastIn;
   return lastIn + inbuf;
}

bool BufferedStream::readString( String &target, uint32 size )
{
   byte *target_buffer;

   if ( target.allocated() >= size )
      target_buffer = target.getRawStorage();
   else {
      target_buffer = (byte *) memAlloc( size );
      if ( target_buffer == 0 )
      {
         setError( -1 );
         return 0;
      }
   }

   int32 sret = this->read( target_buffer, size );

   if ( sret >= 0 )
   {
      target.adopt( (char *) target_buffer, sret, sret + 1 );
   }

   return sret >= 0;
}


int32 BufferedStream::write( const void *buf, int32 size )
{
   if ( size <= 0 )
      return 0;

   byte* buffer = (byte*) buf;
   if ( size + m_bufPos > m_bufSize )
   {
      // flush up to current position, if we have something to flush.
      if ( m_bufPos > 0 && m_changed )
      {
         m_bufLen = m_bufPos; // no concern in truncating, as we'd overwrite
         if ( ! flush() )
            return -1;
      }

      // then write directly the rest.
      int moved = FileStream::write( buffer, size );

      // reset the pointer
      m_bufPos = m_bufLen = 0;

      if ( moved >= 0 ) {
         m_lastMoved = moved + m_bufPos;
         return m_lastMoved;
      }
      else
         return -1;

   }

   // else, we write in the buffer.
   m_lastMoved = size;
   if ( size == 1 )
      m_buffer[ m_bufPos ] = *buffer;
   else
      memcpy( m_buffer + m_bufPos, buffer, size );
   m_bufPos += size;

   // eventually, we enlarge the buffer.
   if ( m_bufPos > m_bufLen )
      m_bufLen = m_bufPos;

   m_changed = true;

   return size;

}

bool BufferedStream::writeString( const String &content, uint32 begin, uint32 end )
{
   uint32 charSize = content.manipulator()->charSize();
   uint32 start = begin * charSize;
   uint32 stop = content.size();
   if ( end < stop / charSize )
      stop = end * charSize;

   return this->write( content.getRawStorage() + start, stop - start ) > 0;
}

bool BufferedStream::close()
{
   flush();
   return FileStream::close();
}

int64 BufferedStream::tell()
{
   if ( m_bufLen > 0 )
      return m_filePos + m_bufPos;
   else
      return FileStream::tell();
}

bool BufferedStream::truncate( int64 pos )
{
   m_bufLen = m_bufPos;
   if ( flush() )
      return false;

   if ( pos == -1 )
      seekBegin( m_filePos + pos );

   return FileStream::truncate( pos );
}

int32 BufferedStream::readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData )
{
   if ( m_bufPos < m_bufLen )
      return true;

   return FileStream::readAvailable( msecs_timeout, sysData );
}

int32 BufferedStream::writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData )
{
   if ( m_bufPos < m_bufSize )
      return true;

   return FileStream::writeAvailable( msecs_timeout, sysData );
}

int64 BufferedStream::seek( int64 pos, e_whence whence )
{
   flush();
   m_bufLen = m_bufPos = 0;

   return FileStream::seek( pos, whence );
}

bool BufferedStream::flush()
{
   if ( ! m_changed || m_bufLen == 0 )
      return true;

   return FileStream::write( m_buffer, m_bufLen ) >= 0;
}

}


/* end of bufferedstream.cpp */
