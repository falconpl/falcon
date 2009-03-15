/*
   FALCON - The Falcon Programming Language.
   FILE: smembuf.cpp

   Shared memory buffer type redefinition.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Apr 2008 17:04:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/vm.h>
#include <falcon/stream.h>
#include "smembuf.h"

namespace Falcon {

SharedMemBuf::SharedMemBuf( byte *data, uint32 size, int *rc, Sys::Mutex *mtx ):
   MemBuf( size, data, size, false ),
   m_refCountPtr( rc ),
   m_mutex( mtx )
{
}

void SharedMemBuf::incref()
{
   m_mutex->lock();
   (*m_refCountPtr)++;
   m_mutex->unlock();
}

void SharedMemBuf::decref( byte *m_data )
{
   m_mutex->lock();
   bool bDestroy = --(*m_refCountPtr) == 0;
   m_mutex->unlock();

   if( bDestroy )
   {
      memFree( m_data );
      memFree( m_refCountPtr );
      delete m_mutex;
   }
}


bool SharedMemBuf::serialize( Stream *stream, bool bLive ) const
{
   // if not live, fallback to standard membuf serializer
   if ( ! bLive )
      return MemBuf::serialize( stream, false );

   // increment reference and serialize memory
   const_cast< SharedMemBuf *>(this)->incref();

   // write our live deserializer
   MemBuf *(*funcptr)( Stream *stream ) = SharedMemBuf::deserialize;
   stream->write( &funcptr, sizeof( funcptr ) );

   // write our private data
   uint32 wSize = endianInt32( wordSize() );
   stream->write( &wSize, sizeof( wSize ) );
   if ( ! stream->good() ) return false;
   wSize = endianInt32( m_length );
   stream->write( &wSize, sizeof( wSize ) );
   if ( ! stream->good() ) return false;

   // write our reference counter and mutex pointers
   stream->write( &m_refCountPtr, sizeof( m_refCountPtr ) );
   stream->write( &m_mutex, sizeof( m_mutex ) );

   // write a pointer to our memory
   stream->write( &m_memory, sizeof( m_memory ) );

   return stream->good();
}

MemBuf* SharedMemBuf::deserialize( Stream *stream )
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
   nSize = endianInt32( nSize );

   int *refcount;
   Sys::Mutex *mtx;
   byte *mem;

   if ( stream->read( &refcount, sizeof( refcount ) ) != sizeof(refcount) )
      return 0;

   if ( stream->read( &mtx, sizeof( mtx ) ) != sizeof(mtx) )
      return 0;

   if ( stream->read( &mem, sizeof( mem ) ) != sizeof(mem) )
      return 0;

   switch( nWordSize )
   {
      case 1: return new SharedMemBuf_1( mem, nSize, refcount, mtx );
      case 2: return new SharedMemBuf_2( mem, nSize, refcount, mtx );
      case 3: return new SharedMemBuf_3( mem, nSize, refcount, mtx );
      case 4: return new SharedMemBuf_4( mem, nSize, refcount, mtx );
   }

   return 0; // impossible
}

//==========================================================
// Shared membuf 1
//

SharedMemBuf_1::SharedMemBuf_1( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx ):
   SharedMemBuf( data, size, refcount, mtx )
{
}

SharedMemBuf_1::~SharedMemBuf_1()
{
   decref( m_memory );
}

uint8 SharedMemBuf_1::wordSize() const
{
   return 1;
}

uint32 SharedMemBuf_1::length() const
{
   return m_length;
}

uint32 SharedMemBuf_1::get( uint32 pos ) const
{
   return m_memory[ pos ];
}

void SharedMemBuf_1::set( uint32 pos, uint32 value )
{
   m_memory[pos] = (byte) value;
}


//==========================================================
// Shared membuf 2
//

SharedMemBuf_2::SharedMemBuf_2( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx ):
   SharedMemBuf( data, size, refcount, mtx )
{
}

SharedMemBuf_2::~SharedMemBuf_2()
{
   decref( m_memory );
}

uint8 SharedMemBuf_2::wordSize() const
{
   return 2;
}


uint32 SharedMemBuf_2::length() const
{
   return m_length / 2;
}


uint32 SharedMemBuf_2::get( uint32 pos ) const
{
   return endianInt16(((uint16 *)m_memory)[pos]);
}


void SharedMemBuf_2::set( uint32 pos, uint32 value )
{
   ((uint16 *)m_memory)[pos] = endianInt16((uint16) value);
}


//==========================================================
// Shared membuf 3
//

SharedMemBuf_3::SharedMemBuf_3( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx ):
   SharedMemBuf( data, size, refcount, mtx )
{
}

SharedMemBuf_3::~SharedMemBuf_3()
{
   decref( m_memory );
}


uint8 SharedMemBuf_3::wordSize() const
{
   return 3;
}

uint32 SharedMemBuf_3::length() const
{
   return m_length/3;
}


uint32 SharedMemBuf_3::get( uint32 pos ) const
{
   byte *p = m_memory + (pos * 3);

   // always act as little endian
   return p[0] | p[1] << 8 | p[2] << 16;
}


void SharedMemBuf_3::set( uint32 pos, uint32 value )
{
   byte *p = m_memory + (pos * 3);

   // always act as little endian
   p[0] = value & 0xff;
   p[1] = (value >> 8) & 0xff;
   p[2] = (value >> 16) & 0xff;
}

//==========================================================
// Shared membuf 4
//

SharedMemBuf_4::SharedMemBuf_4( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx ):
   SharedMemBuf( data, size, refcount, mtx )
{
}

SharedMemBuf_4::~SharedMemBuf_4()
{
   decref( m_memory );
}

uint8 SharedMemBuf_4::wordSize() const
{
   return 4;
}

uint32 SharedMemBuf_4::length() const
{
   return m_length/4;
}


uint32 SharedMemBuf_4::get( uint32 pos ) const
{
   return endianInt32(((uint32 *)m_memory)[pos]);
}


void SharedMemBuf_4::set( uint32 pos, uint32 value )
{
   ((uint32 *)m_memory)[pos] = endianInt32( (uint32)value);
}

}


/* end of smembuf.cpp */
