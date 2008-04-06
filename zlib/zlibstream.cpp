/*
  * FALCON - The Falcon Programming Language.
  * FILE: zlibstream.cpp
  *
  * Implementation of ZLibStream.
  * -------------------------------------------------------------------
  * Author: Jeremy Cowgar
  * Begin: Jan 19 2008
  *
  * -------------------------------------------------------------------
  * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
  *
  * See LICENSE file for licensing details.
  */

/**
 * \file
 * Implementation of ZLibStream.
 */

#include "zlibstream.h"

namespace Falcon {

ZLibStream::ZLibStream( int32 size )
   : Stream( t_membuf )
{
}

ZLibStream::ZLibStream( const ZLibStream &zlibstream )
   : Stream( t_membuf )
{
}

ZLibStream::~ZLibStream()
{
}

bool ZLibStream::errorDescription( String &desciption ) const
{
   return false;
}

bool ZLibStream::close()
{
   return false;
}

int32 ZLibStream::read( void *buffer, int32 size )
{
   return -1;
}

bool ZLibStream::readString( String &target, uint32 size )
{
   return false;
}

int32 ZLibStream::write( const void *buffer, int32 size )
{
   return -1;
}

bool ZLibStream::writeString( const String &source, uint32 begin, uint32 end )
{
   return false;
}

bool ZLibStream::put( uint32 chr )
{
   return false;
}

bool ZLibStream::get( uint32 &chr )
{
   return false;
}

int64 ZLibStream::seek( int64 pos, Stream::e_whence w )
{
   return -1;
}

int64 ZLibStream::tell()
{
   return -1;
}

bool ZLibStream::truncate( int64 pos )
{
   return false;
}

int32 ZLibStream::readAvailable( int32 )
{
   return -1;
}

int32 ZLibStream::writeAvailable( int32 )
{
   return -1;
}

int64 ZLibStream::lastError() const
{
   return (int64) m_lastError;
}

UserData *ZLibStream::clone() const
{
   ZLibStream *zstr = new ZLibStream( *this );
   return zstr;
}

}

/* end of zlibstream.cpp */
