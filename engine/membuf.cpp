/*
   FALCON - The Falcon Programming Language.
   FILE: membuf.cpp

   Core memory buffer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 Mar 2008 23:07:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Memory buffer - Pure memory for Falcon.
*/

#include <falcon/membuf.h>
#include <falcon/memory.h>
#include <falcon/stream.h>
#include <falcon/common.h>

namespace Falcon {

MemBuf::MemBuf( VMachine *vm, uint32 size ):
   Garbageable( vm, sizeof( this ) + size ),
   m_dependant(0)
{
   m_memory = (byte *) memAlloc( size );
   m_size = size;
   m_bOwn = true;
}

MemBuf::MemBuf( VMachine *vm, byte *data, uint32 size, bool bOwn ):
   Garbageable( vm, sizeof( this ) + size ),
   m_memory( data ),
   m_size( size ),
   m_bOwn( bOwn ),
   m_dependant(0)
{
}

MemBuf::~MemBuf()
{
   if ( m_bOwn )
      memFree( m_memory );
}

bool MemBuf::serialize( Stream *stream, bool bLive ) const
{
   if ( bLive )
   {
      // write the live serializer
      MemBuf *(*funcptr)( VMachine *vm, Stream *stream ) = MemBuf::deserialize;
      stream->write( &funcptr, sizeof( funcptr ) );
   }

   uint32 wSize = endianInt32( wordSize() );
   stream->write( &wSize, sizeof( wSize ) );
   if ( ! stream->good() ) return false;
   wSize = endianInt32( m_size );
   stream->write( &wSize, sizeof( wSize ) );
   if ( ! stream->good() ) return false;
   if ( m_size > 0 )
   {
      stream->write( m_memory, m_size );
      if ( ! stream->good() ) return false;
   }

   return true;
}

MemBuf *MemBuf::deserialize( VMachine *vm, Stream *stream )
{
   uint32 nWordSize;
   if ( stream->read( &nWordSize, sizeof( nWordSize ) ) != sizeof( nWordSize ) )
      return 0;
   nWordSize = endianInt32( nWordSize );
   if ( nWordSize < 1 || nWordSize > 4 )
      return 0;

   uint32 nSize;
   if ( stream->read( &nSize, sizeof( nSize ) ) != sizeof( nSize ) )
      return 0;
   nSize = (uint32) endianInt32( nSize );

   byte *mem = (byte *) memAlloc( nSize );
   if ( mem == 0 )
      return 0;

   if ( stream->read( mem, nSize ) != (int32) nSize )
   {
      memFree( mem );
      return 0;
   }

   switch( nWordSize )
   {
      case 1: return new MemBuf_1( vm, mem, nSize, true );
      case 2: return new MemBuf_2( vm, mem, nSize, true );
      case 3: return new MemBuf_3( vm, mem, nSize, true );
      case 4: return new MemBuf_4( vm, mem, nSize, true );
   }

   return 0; // impossible
}

MemBuf *MemBuf::create( VMachine *vm, int bpp, uint32 nSize )
{
   switch( bpp )
   {
      case 1: return new MemBuf_1( vm, nSize );
      case 2: return new MemBuf_2( vm, nSize * 2);
      case 3: return new MemBuf_3( vm, nSize * 3);
      case 4: return new MemBuf_4( vm, nSize * 4);
   }

   return 0;
}


uint8 MemBuf_1::wordSize() const
{
   return 1;
}

uint32 MemBuf_1::length() const
{
   return m_size;
}

uint32 MemBuf_1::get( uint32 pos ) const
{
   return m_memory[ pos ];
}

void MemBuf_1::set( uint32 pos, uint32 value )
{
   m_memory[pos] = (byte) value;
}



uint8 MemBuf_2::wordSize() const
{
   return 2;
}


uint32 MemBuf_2::length() const
{
   return m_size / 2;
}


uint32 MemBuf_2::get( uint32 pos ) const
{
   return endianInt16(((uint16 *)m_memory)[pos]);
}


void MemBuf_2::set( uint32 pos, uint32 value )
{
   ((uint16 *)m_memory)[pos] = endianInt16((uint16) value);
}



uint8 MemBuf_3::wordSize() const
{
   return 3;
}

uint32 MemBuf_3::length() const
{
   return m_size/3;
}


uint32 MemBuf_3::get( uint32 pos ) const
{
   byte *p = m_memory + (pos * 3);

   // always act as little endian
   return p[0] | p[1] << 8 | p[2] << 16;
}


void MemBuf_3::set( uint32 pos, uint32 value )
{
   byte *p = m_memory + (pos * 3);

   // always act as little endian
   p[0] = value & 0xff;
   p[1] = (value >> 8) & 0xff;
   p[2] = (value >> 16) & 0xff;
}



uint8 MemBuf_4::wordSize() const
{
   return 4;
}

uint32 MemBuf_4::length() const
{
   return m_size/4;
}


uint32 MemBuf_4::get( uint32 pos ) const
{
   return endianInt32(((uint32 *)m_memory)[pos]);
}


void MemBuf_4::set( uint32 pos, uint32 value )
{
   ((uint32 *)m_memory)[pos] = endianInt32( (uint32)value);
}


}

/* end of membuf.cpp */
