/*
   FALCON - The Falcon Programming Language
   FILE: stream.cpp

   Implementation of common stream utility functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of common stream utility functions
*/

#include <falcon/stream.h>
#include <falcon/memory.h>

#include <string.h>

namespace Falcon {

Stream::Stream( const Stream &other ):
   m_rhBufferSize( other.m_rhBufferSize ),
   m_rhBufferPos( other.m_rhBufferPos ),
   m_streamType( other.m_streamType ),
   m_status( other.m_status ),
   m_lastMoved( other.m_lastMoved )
{
   if ( m_rhBufferSize != 0 )
   {
      m_rhBuffer = (uint32 *) memAlloc( m_rhBufferSize * sizeof( uint32 ) );
      memcpy( m_rhBuffer, other.m_rhBuffer, m_rhBufferSize * sizeof( uint32 ) );
   }
   else
      m_rhBuffer = 0;
}

bool Stream::errorDescription( ::Falcon::String &description ) const
{
   if ( m_status == t_unsupported ) {
      description = "Unsupported operation for this stream";
      return true;
   }

	return false;
}

Stream::~Stream()
{
   if ( m_rhBuffer != 0 )
      memFree( m_rhBuffer );
}

//=================
// Private members

void Stream::pushBuffer( uint32 chr )
{
   if ( m_rhBufferPos == m_rhBufferSize )
   {
      m_rhBufferSize += FALCON_READAHEAD_BUFFER_BLOCK;
      uint32 *buf = (uint32 *) memRealloc( m_rhBuffer, m_rhBufferSize *sizeof(uint32) );
      m_rhBuffer = buf;
   }
   m_rhBuffer[ m_rhBufferPos ] = chr;
   m_rhBufferPos ++;
   reset();
}

bool Stream::popBuffer( uint32 &chr )
{
   if( m_rhBufferPos == 0 )
      return false;
   m_rhBufferPos--;
   chr = m_rhBuffer[ m_rhBufferPos ];
   return true;
}

//======================
// Public members


void Stream::unget( const String &target )
{
   uint32 pos = target.length();
   while( pos > 0 )
   {
      pos--;
      unget( target.getCharAt( pos ) );
   }
}

bool Stream::readAhead( uint32 &chr )
{
   if ( ! get( chr ) )
      return false;

   unget( chr );
   return true;
}

bool Stream::readAhead( String &target, uint32 size )
{
   if ( readString( target, size ) )
      return false;
   unget( target );
   return true;
}

void Stream::discardReadAhead( uint32 count )
{
   if( count == 0 || count >= m_rhBufferPos )
      m_rhBufferPos = 0;
   else
      m_rhBufferPos -= count;
}

bool Stream::flush()
{
   // does nothing
   return true;
}

bool Stream::readString( String &target, uint32 size )
{
   if ( size == 0 )
      return true;

   uint32 chr;

   // if we can't get EVEN a char, return false.
   if( ! get( chr ) || ! good() )
      return false;

   target.append( chr );
   size --;
   while( size > 0 )
   {
      // if we can't get a char, return false on stream error.
      if ( ! get( chr ) )
         return good();

      target.append( chr );
      size --;
   }

   return true;
}

bool Stream::writeString( const String &source, uint32 begin, uint32 end )
{
   uint32 pos = begin;
   if ( end > source.length() )
      end = source.length();

   while( pos < end ) {
      // some error in writing?
      if ( ! put( source.getCharAt( pos ) ) )
         return false;

      ++pos;
   }

   return true;
}




//======================================
// Overridables
//

Stream *Stream::clone() const
{
   return 0;
}

bool Stream::close()
{
   status( t_unsupported );
   return false;
}


int32 Stream::read( void *, int32 )
{
   status( t_unsupported );
   return -1;
}


int32 Stream::write( const void *, int32 )
{
   status( t_unsupported );
   return -1;
}


int64 Stream::tell()
{
   status( t_unsupported );
   return -1;
}


bool Stream::truncate( int64 )
{
   status( t_unsupported );
   return false;
}


int32 Stream::readAvailable( int32, const Sys::SystemData *sysData )
{
   status( t_unsupported );
   return -1;
}


int32 Stream::writeAvailable( int32, const Sys::SystemData *sysData )
{
   status( t_unsupported );
   return -1;
}

bool Stream::put( uint32 chr )
{
   status( t_unsupported );
   return false;
}


int64 Stream::seek( int64 , e_whence )
{
   status( t_unsupported );
   return -1;
}


int64 Stream::lastError() const
{
   return -1;
}


bool Stream::get( uint32 &chr )
{
   status( t_unsupported );
   return false;
}

}


/* end of stream.cpp */
