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
#include <falcon/vm.h>

namespace Falcon {

MemBuf::MemBuf( uint32 ws, uint32 length ):
   Garbageable(),
   m_length( length ),
   m_mark( INVALID_MARK ),
   m_limit( length ),
   m_position( 0 ),
   m_wordSize( ws ),
   m_byteOrder(0xFEFF),
   m_dependant(0)
{
   m_memory = (byte *) memAlloc( length * ws );
   m_bOwn = true;
}

MemBuf::MemBuf( uint32 ws, byte *data, uint32 length, bool bOwn ):
   Garbageable(),
   m_memory( data ),
   m_length( length ),
   m_mark( INVALID_MARK ),
   m_limit( length ),
   m_position( 0 ),
   m_wordSize( ws ),
   m_dependant(0),
   m_bOwn( bOwn )
{
}

MemBuf::~MemBuf()
{
   if ( m_bOwn && m_memory != 0 )
      memFree( m_memory );
}

void MemBuf::setData( byte *data, uint32 size, bool bOwn )
{
   if ( m_bOwn && m_memory != 0 )
      memFree( m_memory );

   m_memory = data;
   m_length = size/m_wordSize;
   m_bOwn = bOwn;
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
   wSize = endianInt32( m_length );
   stream->write( &wSize, sizeof( wSize ) );
   int16 ws = endianInt16( m_wordSize );
   stream->write( &ws, sizeof( ws ) );
   if ( ! stream->good() ) return false;
   if ( m_length > 0 )
   {
      stream->write( m_memory, m_length * m_wordSize );
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

   uint32 nWS;
   if ( stream->read( &nWS, sizeof( nWS ) ) != sizeof( nWS ) )
      return 0;

   uint32 nSize;
   if ( stream->read( &nSize, sizeof( nSize ) ) != sizeof( nSize ) )
      return 0;
   nSize = (uint32) endianInt32( nSize );

   byte *mem = (byte *) memAlloc( nSize * nWS );
   if ( mem == 0 )
      return 0;

   if ( stream->read( mem, nSize*nWS ) != (int32) (nSize*nWS) )
   {
      memFree( mem );
      return 0;
   }

   switch( nWordSize )
   {
      case 1: return new MemBuf_1( mem, nSize, true );
      case 2: return new MemBuf_2( mem, nSize, true );
      case 3: return new MemBuf_3( mem, nSize, true );
      case 4: return new MemBuf_4( mem, nSize, true );
   }

   return 0; // impossible
}

MemBuf *MemBuf::create( VMachine *vm, int bpp, uint32 nSize )
{
   switch( bpp )
   {
      case 1: return new MemBuf_1( nSize );
      case 2: return new MemBuf_2( nSize );
      case 3: return new MemBuf_3( nSize );
      case 4: return new MemBuf_4( nSize );
   }

   return 0;
}


uint32 MemBuf_1::get( uint32 pos ) const
{
   return m_memory[ pos ];
}

void MemBuf_1::set( uint32 pos, uint32 value )
{
   m_memory[pos] = (byte) value;
}



uint32 MemBuf_2::get( uint32 pos ) const
{
   return endianInt16(((uint16 *)m_memory)[pos]);
}


void MemBuf_2::set( uint32 pos, uint32 value )
{
   ((uint16 *)m_memory)[pos] = endianInt16((uint16) value);
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


uint32 MemBuf_4::get( uint32 pos ) const
{
   return endianInt32(((uint32 *)m_memory)[pos]);
}


void MemBuf_4::set( uint32 pos, uint32 value )
{
   ((uint32 *)m_memory)[pos] = endianInt32( (uint32)value);
}

void MemBuf::readProperty( const String &prop, Item &item )
{
   VMachine *vm = VMachine::getCurrent();
   fassert( vm != 0 );

   // try to find a generic method
   CoreClass* cc = vm->getMetaClass( FLC_ITEM_MEMBUF );
   if ( cc != 0 )
   {
      uint32 id;
      if( cc->properties().findKey( prop, id ) )
      {
         item = *cc->properties().getValue( id );
         item.methodize( this );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
}

void MemBuf::writeProperty( const String &prop, const Item &item )
{
   throw new AccessError( ErrorParam( e_prop_ro, __LINE__ ).extra( prop ) );
}

void MemBuf::readIndex( const Item &index, Item &target )
{
   switch( index.type() )
   {
      case FLC_ITEM_INT:
      {
         int64 pos = (int64) index.asInteger();
         uint32 uPos = (uint32) (pos >= 0 ? pos : length() + pos);
         if ( uPos < length() )
         {
            target = (int64) get( uPos );
            return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         int64 pos = (int64) index.asNumeric();
         uint32 uPos = (uint32) (pos >= 0 ? pos : length() + pos);
         if ( uPos < length() )  {
            target = get( uPos );
            return;
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         readIndex( index.asReference()->origin(), target );
         return;
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "LDV" ) );
}

void MemBuf::writeIndex( const Item &index, const Item &target )
{
   uint32 data;
   switch( target.type() )
   {
      case FLC_ITEM_INT: data = (uint32) target.asInteger(); break;
      case FLC_ITEM_NUM: data = (uint32) target.asNumeric(); break;
      case FLC_ITEM_REFERENCE:
         writeIndex( index, target.asReference()->origin() );
         return;

      default:
         throw new TypeError( ErrorParam( e_param_type, __LINE__ ).extra( "STV" ) );
   }

   switch( index.type() )
   {
      case FLC_ITEM_INT:
      {
         int64 pos = (int64) index.asInteger();
         uint32 uPos = (uint32) (pos >= 0 ? pos : length() + pos);
         if ( uPos < length() )
         {
            set( uPos, data );
            return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         int64 pos = (int64) index.asNumeric();
         uint32 uPos = (uint32) (pos >= 0 ? pos : length() + pos);
         if ( uPos < length() )  {
            set( uPos, data );
            return;
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         writeIndex( index.asReference()->origin(), target );
         return;
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "STV" ) );
}

}

/* end of membuf.cpp */
