/*
   FALCON - The Falcon Programming Language
   FILE: rosstream.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 19 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/rosstream.h>
#include <cstring>
#include <string.h>
namespace Falcon {

ROStringStream::ROStringStream( const String &source ):
   StringStream( -1 )
{
   m_b->m_membuf = const_cast< byte *>( source.getRawStorage() );
   m_b->m_length = source.size();
   m_b->m_allocated = source.size();
   m_b->m_pos = 0;
   m_b->m_lastError = 0;
}

ROStringStream::ROStringStream( const char *source, int size ):
   StringStream( -1 )
{
   m_b->m_membuf = (byte *) const_cast< char *>( source );
   m_b->m_length = size == -1 ? strlen( source ) : size;
   m_b->m_allocated = m_b->m_length;
   m_b->m_pos = 0;
   m_b->m_lastError = 0;
}

ROStringStream::ROStringStream( const ROStringStream& other ):
   StringStream( -1 )
{
   m_b->m_membuf = other.m_b->m_membuf;
   m_b->m_length = other.m_b->m_length;
   m_b->m_allocated = other.m_b->m_allocated;
   m_b->m_pos = other.m_b->m_pos;
   m_b->m_lastError = other.m_b->m_lastError;
}

bool ROStringStream::close()
{
   if( m_b->m_membuf != 0 ) {
      m_b->m_allocated = 0;
      m_b->m_length = 0;
      m_b->m_membuf = 0;
      status( t_none );
      return true;
   }
   return false;
}

int32 ROStringStream::write( const void *buffer, int32 size )
{
   status( t_unsupported );
   return -1;
}

int32 ROStringStream::write( const String &source )
{
   status( t_unsupported );
   return -1;
}

int32 ROStringStream::writeAvailable( int32 msecs, const Falcon::Sys::SystemData* )
{
   status( t_unsupported );
   return -1;
}

bool ROStringStream::truncate( int64 pos )
{
   status( t_unsupported );
   return false;
}

FalconData *ROStringStream::clone() const
{
   return new ROStringStream( *this );
}

}


/* end of rosstream.cpp */
