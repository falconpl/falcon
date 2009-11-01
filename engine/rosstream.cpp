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
   setBuffer( source );
}

ROStringStream::ROStringStream( const char *source, int size ):
   StringStream( -1 )
{
   setBuffer( source, size );
}

ROStringStream::ROStringStream( const ROStringStream& other ):
   StringStream( other )
{
}

bool ROStringStream::close()
{
   return detachBuffer();
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

ROStringStream *ROStringStream::clone() const
{
   return new ROStringStream( *this );
}

}


/* end of rosstream.cpp */
