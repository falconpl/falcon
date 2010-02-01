/*
   FALCON - The Falcon Programming Language.
   FILE: string.cpp

   Implementation of Core Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer nov 24 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of Core Strings.
   \todo Add support for intenrational strings.
*/

#include <falcon/memory.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/common.h>
#include <falcon/vm.h>
#include <string.h>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

namespace Falcon {

namespace csh {

Static handler_static;
Buffer handler_buffer;
Static16 handler_static16;
Buffer16 handler_buffer16;
Static32 handler_static32;
Buffer32 handler_buffer32;


template<typename t1, typename t2>
inline void copySized( byte* dest_, byte* src_, uint32 size )
{
   if ( size != 0 )
   {
      t1* dest = (t1*) dest_;
      t2* src = (t2*) src_;

      do {
         size--;
         dest[size] = (t1) src[size];
      } while( size > 0 );
   }
}

template<typename T>
inline void copySame( byte* dest_, byte* src_, uint32 size )
{
   if ( size != 0 )
   {
      uint32 len = size * sizeof(T);
      T* dest = (T*) dest_;
      T* src = (T*) src_;


      // overlapping? -- use memmove, else use memcpy; but do the check only if worth
      if( len < 10 ||
            (src + len > dest) ||
            (dest + len > src )
      )
         memmove( dest, src, len );
      else
         memcpy( dest, src, len );
   }
}

// service function; adapts a smaller buffer into a larger one
// srclen is in "elements" (bytes * charLen).
// returns also the dynamic buffer manipulator useful to handle the target buffer
static Base* adaptBuffer( byte *srcBuffer, uint32 srcPos, uint32 srcCharLen,
                          byte *destBuffer, uint32 destPos, uint32 destCharLen,
                          uint32 srcLen )
{
   srcBuffer += srcPos * srcCharLen;
   destBuffer += destPos * destCharLen;

   switch( destCharLen )
   {
      case 1:
         switch( srcCharLen ) {
            case 1: copySame<byte>( destBuffer, srcBuffer, srcLen ); break;
            case 2: copySized<byte, uint16>( destBuffer, srcBuffer, srcLen ); break;
            case 4: copySized<byte, uint32>( destBuffer, srcBuffer, srcLen ); break;
         }
         return &handler_buffer;

      case 2:
         switch( srcCharLen ) {
            case 1: copySized<uint16, byte>( destBuffer, srcBuffer, srcLen ); break;
            case 2: copySame<uint16>( destBuffer, srcBuffer, srcLen ); break;
            case 4: copySized<uint16, uint32>( destBuffer, srcBuffer, srcLen ); break;
         }
         return &handler_buffer16;

      case 4:
         switch( srcCharLen ) {
            case 1: copySized<uint32,byte>( destBuffer, srcBuffer, srcLen ); break;
            case 2: copySized<uint32,uint16>( destBuffer, srcBuffer, srcLen ); break;
            case 4: copySame<uint32>( destBuffer, srcBuffer, srcLen ); break;
         }
         return &handler_buffer32;
   }

   return 0;
}

uint32 Byte::length( const String *str ) const
{
   return str->size() / charSize();
}


uint32 Byte::getCharAt( const String *str, uint32 pos ) const
{
   return (uint32) str->getRawStorage()[pos];
}

uint32 Static16::length( const String *str ) const
{
   return str->size() >> 1;
}

uint32 Static32::length( const String *str ) const
{
   return str->size() >> 2;
}

uint32 Buffer16::length( const String *str ) const
{
   return str->size() >> 1;
}

uint32 Buffer32::length( const String *str ) const
{
   return str->size() >> 2;
}


uint32 Static16::getCharAt( const String *str, uint32 pos ) const
{
   return (uint32)  reinterpret_cast< uint16 *>(str->getRawStorage())[ pos ];
}

uint32 Buffer16::getCharAt( const String *str, uint32 pos ) const
{
   return (uint32)  reinterpret_cast< uint16 *>(str->getRawStorage())[ pos ];
}

uint32 Static32::getCharAt( const String *str, uint32 pos ) const
{
   return reinterpret_cast< uint32 *>(str->getRawStorage())[ pos ];
}

uint32 Buffer32::getCharAt( const String *str, uint32 pos ) const
{
   return reinterpret_cast< uint32 *>(str->getRawStorage())[ pos ];
}



void Byte::subString( const String *str, int32 start, int32 end, String *tgt ) const
{
   uint32 nlen = str->length();
   
   if( start < 0 )
      start = int(nlen) + start;
   if( end < 0 )
      end = int(nlen) + end + 1;
   if ( start < 0 || start >= (int)nlen || end < 0 || end == start) {
      tgt->size( 0 );
      return;
   }


   byte *storage, *source;
   int16 cs = charSize();
   source = str->getRawStorage();

   if ( end < start ) {
      uint32 len = start - end + 1;
      if ( tgt->allocated() < len * cs ) {
         storage = (byte *) memAlloc( len * cs );
      }
      else
         storage = tgt->getRawStorage();

      switch( cs )
      {
         case 1:
         {
            for( uint32 i = 0; i < len ; i ++ )
               storage[i] = source[start-i];
            tgt->size( len );
         }
         break;

         case 2:
         {
            uint16 *storage16 = (uint16 *) storage;
            uint16 *source16 = (uint16 *) source;
            for( uint32 i = 0; i < len ; i ++ )
               storage16[i] = source16[start-i];
            tgt->size( len * 2 );
         }
         break;

         case 4:
         {
            uint32 *storage32 = (uint32 *) storage;
            uint32 *source32 = (uint32 *) source;
            for( uint32 i = 0; i < len ; i ++ )
               storage32[i] = source32[start-i];
            tgt->size( len * 4 );
         }
         break;
      }
   }
   else {
      if ( end > (int)nlen ) 
         end = nlen;
         
      uint32 len = (end - start)*cs;
      if ( tgt->allocated() < len ) {
         storage = (byte *) memAlloc( len );
      }
      else
         storage = tgt->getRawStorage();

      memcpy( storage, str->getRawStorage() + (start * cs) , len  );
      tgt->size( len );
   }

   // was the storage not enough?
   if ( storage != tgt->getRawStorage() )
   {
      if ( tgt->allocated() != 0 )
         tgt->manipulator()->destroy( tgt );

      tgt->allocated( tgt->size() );
      tgt->setRawStorage( storage );
   }

   switch( cs )
   {
      case 1: tgt->manipulator( &handler_buffer ); break;
      case 2: tgt->manipulator( &handler_buffer16 ); break;
      case 4: tgt->manipulator( &handler_buffer32 ); break;
   }
}


bool Byte::change( String *str, uint32 start, uint32 end, const String *source ) const
{
   uint32 strLen = str->length();

   if ( start >= strLen )
      return false;

   if ( end > strLen )
      end = strLen;


   if ( end < start ) {
      uint32 temp = end;
      end = start+1;
      start = temp;
   }
   int32 len = end - start;
   insert( str, start, len, source );
   return true;
}


String *Byte::clone( const String *str ) const
{
   return new String( *str );
}

uint32 Byte::find( const String *str, const String *element, uint32 start, uint32 end ) const
{
   if ( str->size() == 0 || element->size() == 0 )
      return npos;

   if ( end > str->length() )  // npos is defined to be greater than any size
      end = str->length();

   if ( end < start ) {
      uint32 temp = end;
      end = start;
      start = temp;
   }

   uint32 pos = start;
   uint32 keyStart = element->getCharAt( 0 );
   uint32 elemLen = element->length();

   while( pos + elemLen <= end ) {
      if ( str->getCharAt( pos ) == keyStart )
      {
         uint32 len = 1;
         while( pos + len < end && len < elemLen && element->getCharAt(len) == str->getCharAt( pos + len ) )
            len++;
         if ( len == elemLen )
            return pos;
      }
      pos++;
   }

   // not found.
   return npos;
}


uint32 Byte::rfind( const String *str, const String *element, uint32 start, uint32 end ) const
{
   if ( str->size() == 0 || element->size() == 0 )
      return npos;

   if ( end > str->length() )  // npos is defined to be greater than any size
      end = str->length();

   if ( end < start ) {
      uint32 temp = end;
      end = start;
      start = temp;
   }

   uint32 keyStart = element->getCharAt( 0 );
   uint32 elemLen = element->length();
   if ( elemLen > (end - start) )
   {
      // can't possibly be found
      return npos;
   }

   uint32 pos = end - elemLen;

   while( pos >= start  ) {
      if ( str->getCharAt( pos ) == keyStart ) {
         uint32 len = 1;
         while( pos + len < end && len < elemLen && element->getCharAt(len) == str->getCharAt( pos + len ) )
            len++;
         if ( len == elemLen )
            return pos;
      }
      if ( pos == 0 ) break;
      pos--;
   }

   // not found.
   return npos;
}


void Byte::remove( String *str, uint32 pos, uint32 len ) const
{
   uint32 sl = str->length();
   if ( len == 0 || pos > sl )
      return;

   uint32 cs = charSize();
   if ( pos + len > sl )
      len = sl - pos;

   uint32 newLen = (sl - len) *cs;
   byte *mem = (byte *) memAlloc( newLen );
   if ( pos > 0 )
      memcpy( mem, str->getRawStorage(), pos * cs );
   if ( pos + len < sl )
      memcpy( mem + pos *cs, str->getRawStorage() +( pos + len) *cs , (sl - pos - len) * cs );

   // for non-static strings...
   if ( str->allocated() != 0 )
   {
      memFree( str->getRawStorage() );
   }
   str->setRawStorage( mem, newLen );

   // subclasses will set correct manipulator if needed.
}


void Byte::bufferize( String *str ) const
{
   // already buffered?
   if ( ! str->isStatic() )
      return;

   uint32 size = str->m_size;
   if ( size != 0 ) {
      uint32 oldSize = str->allocated();

      byte *mem = (byte *) memAlloc( size );
      memcpy( mem, str->getRawStorage(), size );

      if( oldSize != 0 )
      {
         memFree( str->getRawStorage() );
      }

      str->setRawStorage( mem, size );
      str->m_class = str->m_class->bufferedManipulator();
   }
}

void Byte::bufferize( String *str, const String *strOrig ) const
{
   // copy the other string contents.
   if ( str->m_allocated != 0 )
      memFree( str->m_storage );

   uint32 size = strOrig->m_size;
   if ( size == 0 ) {
      str->m_class = &handler_static;
      str->setRawStorage( 0, 0 );
   }
   else {
      byte *mem = (byte *) memAlloc( size );
      memcpy( mem, strOrig->getRawStorage(), size );
      str->setRawStorage( mem, size );
      str->m_class = strOrig->m_class->bufferedManipulator();
   }

}

void Byte::reserve( String *str, uint32 size, bool relative, bool block ) const
{
   if ( relative )
      size += str->m_allocated;

   register int32 chs = charSize();
   if ( block )
   {
      if ( size % FALCON_STRING_ALLOCATION_BLOCK != 0 )
      {
         size /= (FALCON_STRING_ALLOCATION_BLOCK * chs);
         size ++;
         size *= (FALCON_STRING_ALLOCATION_BLOCK * chs);
      }
   }

   uint32 nextAlloc = size * chs;

   // the required size may be already allocated
   if ( nextAlloc > str->allocated() )
   {
      byte *mem = (byte *) memAlloc( nextAlloc );
      uint32 size = str->m_size;
      if ( str->m_size > 0 )
         memcpy( mem, str->m_storage, str->m_size );

      // we can now destroy the old string.
      if ( str->allocated() != 0 )
         memFree( str->m_storage );

      str->m_storage = mem;
      str->m_size = size;
      str->m_allocated = nextAlloc;
   }

   // let the subclasses set the correct manipulator
}


//============================================================0

void Static::shrink( String *str ) const
{
// do nothing
}

void Static::reserve( String *str, uint32 size, bool relative, bool block ) const
{
   Byte::reserve( str, size, relative, block );
   str->m_class = &handler_buffer;
}

const Base *Static::bufferedManipulator() const
{
   return  &handler_buffer;
}

void Static16::reserve( String *str, uint32 size, bool relative, bool block ) const
{
   Byte::reserve( str, size, relative, block );
   str->m_class = &handler_buffer16;
}

const Base *Static16::bufferedManipulator() const
{
   return  &handler_buffer16;
}

void Static32::reserve( String *str, uint32 size, bool relative, bool block ) const
{
   Byte::reserve( str, size, relative, block );
   str->m_class = &handler_buffer32;
}

const Base *Static32::bufferedManipulator() const
{
   return  &handler_buffer32;
}

void Static::setCharAt( String *str, uint32 pos, uint32 chr ) const
{

   byte *buffer;
   int32 size = str->size();

   if( chr <= 0xFF )
   {
      buffer = (byte *) memAlloc( size );
      memcpy( buffer, str->getRawStorage(), size );
      buffer[ pos ] = (byte) chr;
      str->manipulator( &handler_buffer );
   }
   else if ( chr <= 0xFFFF )
   {
      uint16 *buf16 =  (uint16 *) memAlloc( size * 2 );
      buffer = str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf16[ i ] = (uint16) buffer[ i ];

      buf16[ pos ] = (uint16) chr;
      buffer = (byte *) buf16;
      size *= 2;
      str->manipulator( &handler_buffer16 );
   }
   else
   {
      uint32 *buf32 =  (uint32 *) memAlloc( size * 4 );
      buffer = str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf32[ i ] = (uint32) buffer[ i ];

      buf32[ pos ] = chr;
      buffer = (byte *) buf32;
      size *= 4;
      str->manipulator( &handler_buffer32 );
   }

   uint32 oldSize = str->allocated();
   if( oldSize != 0 )
      memFree( str->getRawStorage() );
   str->setRawStorage( buffer, size );
}


void Static16::setCharAt( String *str, uint32 pos, uint32 chr ) const
{

   byte *buffer;
   int32 size = str->size();

   if ( chr <= 0xFFFF )
   {
      uint16 *buf16 =  (uint16 *) memAlloc( size );
      memcpy( buf16, str->getRawStorage(), size );
      buf16[ pos ] = (uint16) chr;
      buffer = (byte *) buf16;
      str->manipulator( &handler_buffer16 );
   }
   else
   {
      uint32 *buf32 =  (uint32 *) memAlloc( size * 2 );
      uint16 *buf16 = (uint16 *) str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf32[ i ] = (uint32) buf16[ i ];

      buf32[ pos ] = chr;
      buffer = (byte *) buf32;
      size *= 2;
      str->manipulator( &handler_buffer32 );
   }

   uint32 oldSize = str->allocated();
   str->setRawStorage( buffer, size );
   if( oldSize != 0 )
      memFree( str->getRawStorage() );
}

void Static32::setCharAt( String *str, uint32 pos, uint32 chr ) const
{
   byte *buffer;
   int32 size = str->size();

   uint32 *buf32 =  (uint32 *) memAlloc( size );
   memcpy( buf32, str->getRawStorage(), size );

   buf32[ pos ] = chr;
   buffer = (byte *) buf32;
   str->manipulator( &handler_buffer32 );
   uint32 oldSize = str->allocated();
   str->setRawStorage( buffer, size );
   if( oldSize != 0 )
      memFree( str->getRawStorage() );
}

void Buffer::setCharAt( String *str, uint32 pos, uint32 chr ) const
{
   byte *buffer;
   int32 size = str->size();

   if( chr <= 0xFF )
   {
      str->getRawStorage()[ pos ] = (byte) chr;
   }
   else if ( chr <= 0xFFFF )
   {
      uint16 *buf16 =  (uint16 *) memAlloc( size * 2 );
      buffer = str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf16[ i ] = (uint16) buffer[ i ];

      buf16[ pos ] = (uint16) chr;
      size *= 2;
      str->manipulator( &handler_buffer16 );
      if( str->allocated() > 0 )
         memFree( buffer );
      str->setRawStorage( (byte *) buf16, size );
   }
   else
   {
      uint32 *buf32 =  (uint32 *) memAlloc( size * 4 );
      buffer = str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf32[ i ] = (uint32) buffer[ i ];

      buf32[ pos ] = chr;
      size *= 4;
      str->manipulator( &handler_buffer32 );
      if( str->allocated() > 0 )
         memFree( buffer );
      str->setRawStorage( (byte *) buf32, size );
   }
}


void Buffer16::setCharAt( String *str, uint32 pos, uint32 chr ) const
{
   if ( chr <= 0xFFFF )
   {
      uint16 *buf16 =  (uint16 *) str->getRawStorage();
      buf16[ pos ] = (uint16) chr;
   }
   else
   {
      int32 size = str->size();
      uint32 *buf32 =  (uint32 *) memAlloc( size * 2 );
      uint16 *buf16 = (uint16 *) str->getRawStorage();
      for ( int i = 0; i < size; i ++ )
         buf32[ i ] = (uint32) buf16[ i ];

      buf32[ pos ] = chr;
      size *= 2;
      str->manipulator( &handler_buffer32 );
      if( str->allocated() > 0 )
         memFree( buf16 );
      str->setRawStorage( (byte *) buf32, size );
   }

}

void Buffer32::setCharAt( String *str, uint32 pos, uint32 chr ) const
{
   uint32 *buf32 = (uint32 *) str->getRawStorage();
   buf32[ pos ] = chr;
}


void Static::insert( String *str, uint32 pos, uint32 len, const String *source ) const
{
   uint32 sourceLen = source->length();

   uint32 strLen = str->length();

   if ( pos + len > str->size() )
      len = str->size() - pos;

   uint32 strCharSize = str->manipulator()->charSize();

   uint32 destCharSize = source->manipulator()->charSize() > str->manipulator()->charSize() ?
      source->manipulator()->charSize() : str->manipulator()->charSize() ; // can be 1 or larger

   uint32 finalSize = destCharSize * (strLen - len + sourceLen );
   uint32 finalAlloc = ((finalSize / FALCON_STRING_ALLOCATION_BLOCK) + 1) *
      FALCON_STRING_ALLOCATION_BLOCK;

   // we know we have to relocate, so just do the relocation step
   byte *mem = (byte*) memAlloc( finalAlloc );
   if ( pos > 0 )
      adaptBuffer( str->getRawStorage(), 0, strCharSize, mem, 0, destCharSize, pos );

   str->manipulator(
      adaptBuffer( source->getRawStorage(), 0, source->manipulator()->charSize(),
                   mem, pos, destCharSize, sourceLen ) );

   if ( pos + len < strLen )
      adaptBuffer( str->getRawStorage(), pos + len, strCharSize,
                   mem,  pos + sourceLen, destCharSize,
                   strLen - pos - len );

   str->size( finalSize );

   // Static strings CAN have non-static memory: expecially if they are de-serialized strings in modules.
   uint32 oldSize = str->allocated();
   str->allocated( finalAlloc );

   if ( oldSize > 0 )
   {
      memFree( str->getRawStorage() );
   }
   str->setRawStorage( mem );
}


void Buffer::insert( String *str, uint32 pos, uint32 len, const String *source ) const
{
   uint32 sourceLen = source->length();

   uint32 strLen = str->length();
   if ( pos + len > str->size() )
      len = str->size() - pos;

   uint32 strCharSize = str->manipulator()->charSize();
   uint32 posBytes = pos *strCharSize;
   uint32 lenBytes = len *strCharSize;

   uint32 destCharSize = source->manipulator()->charSize() > strCharSize ?
      source->manipulator()->charSize() : strCharSize; // can be 1 or larger

   uint32 finalSize = destCharSize * (strLen - len + sourceLen );

   // should we re-allocate?
   if( finalSize > str->allocated() || destCharSize > strCharSize )
   {

      uint32 finalAlloc = ((finalSize / FALCON_STRING_ALLOCATION_BLOCK) + 1) *
         FALCON_STRING_ALLOCATION_BLOCK;

      // we know we have to relocate, so just do the relocation step
      byte *mem = (byte*) memAlloc( finalAlloc );
      if ( pos > 0 )
         adaptBuffer( str->getRawStorage(), 0, strCharSize,
                      mem, 0, destCharSize, pos );

      str->manipulator(
         adaptBuffer( source->getRawStorage(), 0, source->manipulator()->charSize(),
                      mem, pos, destCharSize, sourceLen ) );

      if ( pos + len < strLen )
         adaptBuffer( str->getRawStorage(), pos + len, strCharSize,
                      mem, pos+sourceLen, destCharSize,
                      strLen - pos - len );

      if ( str->allocated() != 0 )
         memFree( str->getRawStorage() );

      str->allocated( finalAlloc );
      str->setRawStorage( mem );
   }
   else
   {
      // should we move the tail?
      if ( pos + len < strLen )
      {
         // can we maintain our char size?
         if( destCharSize == strCharSize )
         {
            uint32 sourceLenBytes = sourceLen * destCharSize;
            // then just move the postfix away
            memmove( str->getRawStorage() + posBytes + sourceLenBytes,
                     str->getRawStorage() + posBytes + lenBytes,
                     str->size() - posBytes - lenBytes );
         }
         else {
            // adapt it to the new size.
               adaptBuffer( str->getRawStorage(), pos + len, strCharSize,
                  str->getRawStorage(), pos + sourceLen, destCharSize,
                  strLen - pos - len );
         }
      }

      // adapt the incoming part
      str->manipulator(
         adaptBuffer( source->getRawStorage(), 0, source->manipulator()->charSize(),
                     str->getRawStorage(), pos, destCharSize,
                     sourceLen ) );

      // eventually adapt the head -- adaptBuffer can work on itself
      if ( pos > 0 && destCharSize != strCharSize )
         str->manipulator(
            adaptBuffer( str->getRawStorage(), 0, strCharSize,
                         str->getRawStorage(), 0, destCharSize,
                         pos ) );
   }

   str->size( finalSize );
}


void Static::remove( String *str, uint32 pos, uint32 len ) const
{
   Byte::remove( str, pos, len );
   // changing string type.
   str->manipulator( &handler_buffer );
}

void Static16::remove( String *str, uint32 pos, uint32 len ) const
{
   Byte::remove( str, pos, len );
   // changing string type.
   str->manipulator( &handler_buffer16 );
}

void Static32::remove( String *str, uint32 pos, uint32 len ) const
{
   Byte::remove( str, pos, len );
   // changing string type.
   str->manipulator( &handler_buffer32 );
}


void Static::destroy( String *str ) const
{
   if ( str->allocated() > 0 ) {
      memFree( str->getRawStorage() );
      str->allocated( 0 );
      str->size(0);
   }
}


void Buffer::shrink( String *str ) const
{
   if( str->size() > str->allocated() )
   {
      if ( str->size() == 0 )
      {
         destroy( str );
      }
      else {
         byte *mem = (byte *) memRealloc( str->getRawStorage(), str->size() );
         if ( mem != str->getRawStorage() )
         {
            memcpy( mem, str->getRawStorage(), str->size() );
            str->setRawStorage( mem );
         }
         str->allocated( str->size() );
      }
   }
}

void Buffer::reserve( String *str, uint32 size, bool relative, bool block ) const
{
   Byte::reserve( str, size, relative, block );
   // manipulator is ok
}


void Buffer::destroy( String *str ) const
{
   if ( str->allocated() > 0 ) {
      memFree( str->getRawStorage() );
      str->allocated( 0 );
      str->size(0);
   }
}

} // namespace csh


//=================================================================
// The string class
//=================================================================


String::String( uint32 size ):
   m_class( &csh::handler_buffer ),
   m_bExported( false ),
   m_bCore( false )
{
   m_storage = (byte *) memAlloc( size );
   m_allocated = size;
   m_size = 0;
}

String::String( const char *data ):
   m_class( &csh::handler_static ),
   m_allocated( 0 ),
   m_storage( (byte*) const_cast< char *>(data) ),
   m_bExported( false ),
   m_bCore( false )
{
   m_size = strlen( data );
}

String::String( const char *data, int32 len ):
   m_class( &csh::handler_buffer ),
   m_bExported( false ),
   m_bCore( false )
{
   m_size = len >= 0 ? len : strlen( data );
   m_allocated = (( m_size / FALCON_STRING_ALLOCATION_BLOCK ) + 1 ) * FALCON_STRING_ALLOCATION_BLOCK;
   m_storage = (byte *) memAlloc( m_allocated );
   memcpy( m_storage, data, m_size );
}


String::String( const wchar_t *data ):
   m_allocated( 0 ),
   m_storage( (byte*) const_cast< wchar_t *>(data) ),
   m_bExported( false ),
   m_bCore( false )
{
   if ( sizeof( wchar_t ) == 2 )
      m_class = &csh::handler_static16;
   else
      m_class = &csh::handler_static32;

   uint32 s = 0;
   while( data[s] != 0 )
      ++s;
   m_size = s * sizeof( wchar_t );
}


String::String( const wchar_t *data, int32 len ):
   m_allocated( 0 ),
   m_storage( (byte *) const_cast< wchar_t *>( data ) ),
   m_bExported( false ),
   m_bCore( false )
{
   if ( sizeof( wchar_t ) == 2 )
      m_class = &csh::handler_buffer16;
   else
      m_class = &csh::handler_buffer32;

   if ( len >= 0 )
   {
      m_size = len * sizeof( wchar_t );
   }
   else
   {
      uint32 s = 0;
      while( data[s] != 0 )
         ++s;
      m_size = s * sizeof( wchar_t );
   }

   m_allocated = (( m_size / FALCON_STRING_ALLOCATION_BLOCK ) + 1 ) * FALCON_STRING_ALLOCATION_BLOCK;
   m_storage = (byte *) memAlloc( m_allocated );
   memcpy( m_storage, data, m_size );
}


String::String( const String &other, uint32 begin, uint32 end ):
   m_allocated( 0 ),
   m_size( 0 ),
   m_storage( 0 ),
   m_bExported( false ),
   m_bCore( false )
{
   // by default, copy manipulator
   m_class = other.m_class;

   if ( other.m_allocated == 0 )
   {
      if ( other.m_size == 0 ) {
         m_size = 0;
         m_allocated = 0;
         setRawStorage( 0 );
      }
      else {
         if( begin < end )
         {
            uint32 cs = m_class->charSize();
            setRawStorage( other.getRawStorage() + begin * cs );
            m_size = (end - begin ) * cs;
            if ( m_size > (other.m_size - (begin *cs )) )
               m_size = other.m_size - (begin *cs );
         }
         else {
            // reverse substring, we have to bufferize.
            other.m_class->subString( &other, begin, end, this );
         }
      }
   }
   else
      other.m_class->subString( &other, begin, end, this );
}


void String::copy( const String &other )
{
   if ( m_allocated != 0 )
      m_class->destroy( this );

   m_class = other.m_class;
   m_size = other.m_size;
   m_allocated = other.m_allocated;
   if ( m_allocated > 0 ) {
      m_storage = (byte *) memAlloc( m_allocated );
      if ( m_size > 0 )
         memcpy( m_storage, other.m_storage, m_size );
   }
   else
      m_storage = other.m_storage;
}


String &String::adopt( char *buffer, uint32 size, uint32 allocated )
{
   if ( m_allocated != 0 )
      m_class->destroy( this );

   m_class = &csh::handler_buffer;
   m_size = size;
   m_allocated = allocated;
   m_storage = (byte *) buffer;


   return *this;
}

String &String::adopt( wchar_t *buffer, uint32 size, uint32 allocated )
{
   if ( m_allocated != 0 )
      m_class->destroy( this );

   if ( sizeof( wchar_t ) == 2 )
      m_class = &csh::handler_buffer16;
   else
      m_class = &csh::handler_buffer32;

   m_size = size * sizeof( wchar_t );
   m_allocated = allocated;
   m_storage = (byte *) buffer;

   return *this;
}

int String::compare( const char *other ) const
{
   uint32 pos = 0;
   uint32 len = length();

   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      if ( c1 < (uint32) *other )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > (uint32) *other || (uint32) *other == 0 )
         return 1;

      other ++;
      pos++;
   }

   if ( *other != 0 )
      // other is greater
      return -1;

   // the strings are the same
   return 0;
}

int String::compareIgnoreCase( const char *other ) const
{
   uint32 pos = 0;
   uint32 len = length();

   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      uint32 cmpval = (uint32) *other;
      if ( c1 >=0x41 && c1 <= 0x5A )
         c1 += 0x20;

      if ( cmpval >=0x41 && cmpval <= 0x5A )
         cmpval += 0x20;

      if ( c1 < cmpval )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > cmpval || cmpval == 0 )
         return 1;

      other ++;
      pos++;
   }

   if ( *other != 0 )
      // other is greater
      return -1;

   // the strings are the same
   return 0;
}

int String::compare( const wchar_t *other ) const
{
   uint32 pos = 0;
   uint32 len = length();

   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      if ( c1 < (uint32) *other )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > (uint32) *other || (uint32) *other == 0 )
         return 1;

      other ++;
      pos++;
   }

   if ( *other != 0 )
      // other is greater
      return -1;

   // the strings are the same
   return 0;
}

int String::compareIgnoreCase( const wchar_t *other ) const
{
   uint32 pos = 0;
   uint32 len = length();

   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      uint32 cmpval = (uint32) *other;
      if ( c1 >=0x41 && c1 <= 0x5A )
         c1 += 0x20;

      if ( cmpval >=0x41 && cmpval <= 0x5A )
         cmpval += 0x20;

      if ( c1 < cmpval )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > cmpval || cmpval == 0 )
         return 1;

      other ++;
      pos++;
   }

   if ( *other != 0 )
      // other is greater
      return -1;

   // the strings are the same
   return 0;
}

int String::compare( const String &other ) const
{
   uint32 len1 = length();
   uint32 len2 = other.length();
   uint32 len = len1 > len2 ? len2 : len1;

   uint32 pos = 0;
   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      uint32 c2 = other.getCharAt( pos );

      if ( c1 < c2 )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > c2 )
         return 1;

      pos++;
   }

   // the strings are the same?
   if ( len1 < len2 )
      return -1;
   // return greater also if other is ended; we are NOT ended...
   else if ( len1 > len2 )
      return 1;

   // yes
   return 0;
}


int String::compareIgnoreCase( const String &other ) const
{
   uint32 len1 = length();
   uint32 len2 = other.length();
   uint32 len = len1 > len2 ? len2 : len1;

   uint32 pos = 0;
   while( pos < len )
   {
      uint32 c1 = getCharAt( pos );
      uint32 c2 = other.getCharAt( pos );

      if ( c1 >=0x41 && c1 <= 0x5A )
         c1 += 0x20;

      if ( c2 >=0x41 && c2 <= 0x5A )
         c2 += 0x20;

      if ( c1 < c2 )
         return -1;
      // return greater also if other is ended; we are NOT ended...
      else if ( c1 > c2 )
         return 1;

      pos++;
   }

   // the strings are the same?

   if ( len1 < len2 )
      return -1;
   // return greater also if other is ended; we are NOT ended...
   else if ( len1 > len2 )
      return 1;

   // yes
   return 0;
}

uint32 String::toCString( char *target, uint32 bufsize ) const
{
   uint32 len = length();

   // we already know that the buffer is too small?
    if ( bufsize <= len )
      return npos;

   uint32 pos = 0;
   uint32 done = 0;

   while( done < len && pos < bufsize )
   {
      uint32 chr = getCharAt( done );

      if ( chr < 0x80 )
      {
         target[ pos ++ ] = chr;
      }
      else if ( chr < 0x800 )
      {
         if ( pos + 2 > bufsize )
            return npos;

         target[ pos ++ ] = 0xC0 | ((chr >> 6 ) & 0x1f);
         target[ pos ++ ] = 0x80 | (0x3f & chr);
      }
      else if ( chr < 0x10000 )
      {
         if ( pos + 3 > bufsize )
            return npos;

         target[ pos ++ ] = 0xE0 | ((chr >> 12) & 0x0f );
         target[ pos ++ ] = 0x80 | ((chr >> 6) & 0x3f );
         target[ pos ++ ] = 0x80 | (0x3f & chr);
      }
      else
      {
         if ( pos + 4 > bufsize )
            return npos;

         target[ pos ++ ] = 0xF0 | ((chr >> 18) & 0x7 );
         target[ pos ++ ] = 0x80 | ((chr >> 12) & 0x3f );
         target[ pos ++ ] = 0x80 | ((chr >> 6) & 0x3f );
         target[ pos ++ ] = 0x80 | (0x3f & chr );
      }

      ++done;
   }

   if ( pos >= bufsize )
      return npos;

   target[pos] = '\0';
   return pos;
}

uint32 String::toWideString( wchar_t *target, uint32 bufsize ) const
{
   uint32 len = length();
   if ( bufsize <= len * sizeof( wchar_t ) )
      return npos;

   if ( sizeof( wchar_t ) == 2 ) {
      for( uint32 i = 0; i < len; i ++ )
      {
         uint32 chr = getCharAt( i );
         if ( chr > 0xFFFF )
            target[ i ] = L'?';
         else
            target[ i ] = chr;
      }
   }
   else {
      for( uint32 i = 0; i < len; i ++ )
      {
         target[ i ] = getCharAt( i );
      }
   }
	target[len] = 0;

   return (int32) len;
}

void String::append( const String &str )
{
/*   if ( str.size() < FALCON_STRING_ALLOCATION_BLOCK )
      m_class->reserve( this, FALCON_STRING_ALLOCATION_BLOCK, true, true ); */
   m_class->insert( this, length(), 0, &str );
}

void String::append( uint32 chr )
{
   if ( size() >= allocated() )
   {
      m_class->reserve( this, size() + FALCON_STRING_ALLOCATION_BLOCK, false, true ); // allocates a whole block next
   }

   // reserve forces to buffer, so we have only to extend
   size( size() + m_class->charSize() );
   // and insert the last char:

   setCharAt( length() - 1, chr );
   // if the chr can't fit our size, the whole string will be refitted, including the last
   // empty char that we added above.
}

void String::prepend( uint32 chr )
{
   if ( size() >= allocated() )
   {
      m_class->reserve( this, size() + FALCON_STRING_ALLOCATION_BLOCK, false, true ); // allocates a whole block next
   }

   // reserve forces to buffer, so we have only to extend
   memmove( m_storage + m_class->charSize(), m_storage, size() );
   size( size() + m_class->charSize() );

   // and insert the first char
   setCharAt( 0, chr );
}

void String::uint32ToHex( uint32 number, char *buffer )
{
   uint32 divisor = 0x10000000;
   int pos = 0;

   while( divisor > 0 ) {
      uint32 rest = number / divisor;
      if( rest > 0 || pos > 0 ) {
         buffer[pos] = rest >= 10 ? 'A' + (rest - 10):'0'+rest;
         pos++;
         number -= rest * divisor;
      }
      divisor = divisor >> 4; // divide by 16
   }
   buffer[pos] = 0;
}

void String::escape( String &strout ) const
{
   int len = length();
   int pos = 0;
   strout.m_class->reserve( &strout, len ); // prepare for at least len chars
   strout.size( 0 ); // clear target string

   while( pos < len )
   {
      uint32 chat = getCharAt( pos );
      switch( chat )
      {
         case '"':
            strout += "\\\""; break;
         case '\r': strout += "\\r"; break;
         case '\n': strout += "\\n"; break;
         case '\t': strout += "\\t"; break;
         case '\b': strout += "\\b"; break;
         case '\\': strout += "\\\\"; break;
         default:
            if ( chat < 8 ) {
               char bufarea[12];
               uint32ToHex( chat, bufarea );
               strout += bufarea;
            }
            else{
               strout += chat;
            }
      }
      pos++;
   }

}

void String::escapeFull( String &strout ) const
{
   int len = length();
   int pos = 0;
   strout.m_class->reserve( &strout, len ); // prepare for at least len chars
   strout.size( 0 ); // clear target string

   while( pos < len )
   {
      uint32 chat = getCharAt( pos );
      switch( chat )
      {
         case '"':  strout += "\\\""; break;
         case '\r': strout += "\\r"; break;
         case '\n': strout += "\\n"; break;
         case '\t': strout += "\\t"; break;
         case '\b': strout += "\\b"; break;
         case '\\': strout += "\\\\"; break;
         default:
            if ( chat < 8 || chat > 127 ) {
               char bufarea[12];
               uint32ToHex( chat, bufarea );
               strout += bufarea;
            }
            else{
               strout += chat;
            }
      }
      pos++;
   }
}

void String::unescape()
{
   uint32 len = length();
   uint32 pos = 0;

   while( pos < len )
   {
      uint32 chat = getCharAt( pos );
      if ( chat == (uint32) '\\' )
      {
         // an escape must take place
         uint32 endSub = pos + 1;
         if( endSub == len - 1 )
            return;

         uint32 chnext = getCharAt( endSub );
         uint32 chsub=0;

         switch( chnext )
         {
            case '"':  chsub = (uint32) '"'; break;
            case '\r': chsub = (uint32) '\r'; break;
            case '\n': chsub = (uint32) '\n'; break;
            case '\t': chsub = (uint32) '\t'; break;
            case '\b': chsub = (uint32) '\b'; break;
            case '\\': chsub = (uint32) '\\'; break;
            case '0':
               // parse octal number
               endSub ++;
               chsub = 0;
               // max lenght of octals = 11 chars, + 2 for stubs
               while( endSub < len && endSub - pos < 13 )
               {
                  chnext = getCharAt( endSub );
                  if ( chnext < 0x30 || chnext > 0x37 )
                     break;
                  chsub <<= 3; //*8
                  chsub |= (0x7) & (chnext - 0x30);
                  endSub ++;
               }
            break;

            case 'x':
               // parse exadecimal number
               endSub ++;
               chsub = 0;
               // max lenght of octals = 11 chars, + 2 for stubs
               while( endSub < len && endSub - pos < 13 )
               {
                  chnext = getCharAt( endSub );
                  if ( chnext >= 0x30 && chnext <= 0x39 ) // 0 - 9
                  {
                     chsub <<= 4; //*16
                     chsub |=  chnext - 0x30;
                  }
                  else if( chnext >= 0x41 && chnext <= 0x46 ) // A - F
                  {
                     chsub <<= 4; //*16
                     chsub |=  chnext - 0x41 + 10;
                  }
                  else if( chnext >= 0x61 && chnext <= 0x66 ) // a - f
                  {
                     chsub <<= 4; //*16
                     chsub |=  chnext - 0x61 + 10;
                  }
                  endSub ++;
               }
            break;
         }
         // substitute the char
         setCharAt( pos, chsub );
         // remove the rest
         remove( pos + 1, endSub - pos );
      }

      pos++;
   }
}


void String::serialize( Stream *out ) const
{
   uint32 size = m_bExported ? m_size | 0x80000000 : m_size;
   size = endianInt32( size );

   out->write( (byte *) &size, sizeof(size) );
   if ( m_size != 0 && out->good() )
   {
      byte chars = m_class->charSize();
      out->write( &chars, 1 );
      #ifdef FALCON_LITTLE_ENDIAN
      out->write( m_storage, m_size );
      #else
      // in big endian environ, we have to reverse the code.
      if( chars == 1 )
      {
         out->write( m_storage, m_size );
      }
      else if ( chars == 2 )
      {
         for( int i = 0; i < m_size/2; i ++ )
         {
            uint16 chr = (uint32) endianInt16((uint16) getCharAt( i ) );
            out->write( (byte *) &chr, 2 );
            if (! out->good() )
               return;
         }
      }
      else if ( chars == 4 )
      {
         for( int i = 0; i < m_size/4; i ++ )
         {
            uint32 chr = (uint32) endianInt32( getCharAt( i ) );
            out->write( (byte *) &chr, 4 );
            if (! out->good() )
               return;
         }
      }
      #endif
   }
}


bool String::deserialize( Stream *in, bool bStatic )
{
   uint32 size;

   in->read( (byte *) &size, sizeof( size ) );
   m_size = endianInt32(size);
   m_bExported = (m_size & 0x80000000) == 0x80000000;
   m_size = m_size & 0x7FFFFFFF;

   // if the size of the deserialized string is 0, we have an empty string.
   if ( m_size == 0 )
   {
      // if we had something allocated, we got to free it.
      if ( m_allocated > 0 )
      {
         memFree( m_storage );
         m_storage = 0;
         m_allocated = 0;
      }

      // anyhow, set the handler to static and return.
      manipulator(&csh::handler_static);
      return true;
   }

   if ( in->good() )
   {
      byte chars;
      in->read( &chars, 1 );

      // determine the needed manipulator
      if ( bStatic )
      {
         switch( chars )
         {
            case 1: manipulator( &csh::handler_static ); break;
            case 2: manipulator( &csh::handler_static16 ); break;
            case 4: manipulator( &csh::handler_static32 ); break;
            default: return false;
         }
      }
      else {
         switch( chars )
         {
            case 1: manipulator( &csh::handler_buffer ); break;
            case 2: manipulator( &csh::handler_buffer16 ); break;
            case 4: manipulator( &csh::handler_buffer32 ); break;
            default: return false;
         }
      }


      m_storage = (byte *) memRealloc( m_storage, m_size );
      if( m_storage == 0 )
         return false;

      m_allocated = m_size;

      #ifdef FALCON_LITTLE_ENDIAN
      in->read( m_storage, m_size );
      #else
      // in big endian environ, we have to reverse the code.
      in->read( m_storage, m_size );
      if ( ! in->good() )
         return;

      if ( chars == 2 )
      {
         uint16* storage16 = (uint16*) m_storage;
         for( int i = 0; i < m_size/2; i ++ )
         {
            storage16[i] = (uint16) endianInt16( storage16[i] );
         }
      }
      else if ( chars == 4 )
      {
         uint32* storage32 = (uint32*) m_storage;
         for( int i = 0; i < m_size/4; i ++ )
         {
            storage32[i] = (uint32) endianInt32( storage32[i] );
         }
      }
      #endif
   }

   return true;
}


void String::c_ize()
{
   if ( allocated() <= size() || getCharAt( length() ) != 0 )
   {
      append( 0 );
      size( size() - m_class->charSize() );
   }
}

bool String::setCharSize( uint32 nsize, uint32 subst )
{
   // same size?
   if ( nsize == m_class->charSize() )
      return true;

   // change only the manipulator?
   if( size() == 0 ) {
      m_class->destroy( this ); // dispose anyhow
      allocated(0);
      switch( nsize ) {
         case 1: m_class = &csh::handler_buffer; break;
         case 2: m_class = &csh::handler_buffer16; break;
         case 4: m_class = &csh::handler_buffer32; break;
         default: return false;
      }

      return true;
   }

   if ( nsize != 1 && nsize != 2 && nsize != 4 )
      return false;

   // full change.
   // use allocated to decide re-allocation under new char size.
   byte *mem = getRawStorage();
   uint32 oldcs = m_class->charSize();
   uint32 nalloc = (allocated()/oldcs) * nsize;
   uint32 oldsize = size();
   byte *nmem = (byte*) memAlloc( nalloc );
   csh::Base* manipulator = csh::adaptBuffer( mem, 0, oldcs, nmem, 0, nsize, length() );
   m_class->destroy( this );
   allocated( nalloc );
   size( (oldsize/oldcs)*nsize );
   m_class = manipulator;
   setRawStorage( nmem );

   return true;
}


bool String::parseInt( int64 &target, uint32 pos ) const
{
   uint32 len = length();
   if ( pos >= len )
      return false;

   target = 0;

   bool neg = false;

   uint32 chnext = getCharAt( pos );
   if ( chnext == (uint32) '-' )
   {
      neg = true;
      pos ++;
      if ( pos == len )
         return false;
      chnext = getCharAt( pos );
   }

   // detect overflow
   int64 tgtCopy = target;
   while( chnext >= 0x30 && chnext <= 0x39 )
   {
      if( target < tgtCopy )
         return false;

      tgtCopy = target;
      target *= 10;
      target += chnext - 0x30;
      pos ++;
      if ( pos == len )
         break;
      chnext = getCharAt( pos );
   }

   if ( chnext < 0x30 || chnext > 0x39 )
      return false;

   if (neg)
      target = -target;

   return true;
}

bool String::parseDouble( double &target, uint32 pos ) const
{
   char buffer[64];
   uint32 maxlen = 63;

   uint32 len = length();
   if ( pos >= len )
      return false;

   // if we are single byte string, just copy
   if ( m_class->charSize() == 1 )
   {
      if ( maxlen > len - pos )
         maxlen = len - pos;
      memcpy( buffer, m_storage, maxlen );
      buffer[ maxlen ] = '\0';
   }
   else {

      // else convert to C string
      uint32 bufpos = 0;
      while ( bufpos < maxlen && pos < len )
      {
         uint32 chr = getCharAt( pos );
         if( chr != (uint32) '-' && chr != (uint32) 'e' && chr != (uint32) 'E' &&
                  chr < 0x30 && chr > 0x39 )
            return false;
         buffer[ bufpos ] = (char) chr;
         bufpos ++;
         pos ++;
      }
   }

   // then apply sscanf
   if ( sscanf( buffer, "%lf", &target ) == 1 )
      return true;
   return false;
}


bool String::parseBin( uint64 &target, uint32 pos ) const
{
   uint32 len = length();
   if ( pos >= len )
      return false;
   // parse octal number
   target = 0;
   uint32 endSub = pos;

   // max lenght of binary = 64 chars, + 2 for stubs
   while( endSub < len && (endSub - pos < 64) )
   {
      uint32 chnext = getCharAt( endSub );
      if ( chnext < 0x30 || chnext > 0x31 )
         break;
      target <<= 1; //*2
      target |= (0x1) & (chnext - 0x30);
      endSub ++;
   }

   if( endSub != pos )
      return true;
   return false;
}


bool String::parseOctal( uint64 &target, uint32 pos ) const
{
   uint32 len = length();
   if ( pos >= len )
      return false;
   // parse octal number
   target = 0;
   uint32 endSub = pos;

   // max lenght of octals = 11 chars, + 2 for stubs
   while( endSub < len && (endSub - pos < 26) )
   {
      uint32 chnext = getCharAt( endSub );
      if ( chnext < 0x30 || chnext > 0x37 )
         break;
      target <<= 3; //*8
      target |= (0x7) & (chnext - 0x30);
      endSub ++;
   }

   if( endSub != pos )
      return true;
   return false;
}

bool String::parseHex( uint64 &target, uint32 pos ) const
{
   uint32 len = length();
   if ( pos >= len )
      return false;
   // parse octal number
   target = 0;
   uint32 endSub = pos;

   while( endSub < len && (endSub - pos < 16) )
   {
      uint32 chnext = getCharAt( endSub );
      if ( chnext >= 0x30 && chnext <= 0x39 ) // 0 - 9
      {
         target <<= 4; //*16
         target |=  chnext - 0x30;
      }
      else if( chnext >= 0x41 && chnext <= 0x46 ) // A - F
      {
         target <<= 4; //*16
         target |=  chnext - 0x41 + 10;
      }
      else if( chnext >= 0x61 && chnext <= 0x66 ) // a - f
      {
         target <<= 4; //*16
         target |=  chnext - 0x61 + 10;
      }
      endSub ++;
   }

   if( endSub != pos )
      return true;

   return false;
}

void String::writeNumber( int64 number )
{
   // prepare the buffer
   bool neg;
   char buffer[21];
   uint32 pos = 19;
   buffer[20] = '\0';


   if ( number == 0 )
   {
      buffer[pos] = '0';
   }
   else {
      if ( number < 0 )
      {
         neg = true;
         number = - number;
         if ( number < 0 )
         {
            // is NAN
            append( "NaN" );
            return;
         }
      }
      else
         neg = false;

      while( number != 0 ) {
         buffer[pos--] = (char) ((number % 10) + 0x30);
         number /= 10;
      }

      if ( neg )
         buffer[ pos ] = '-';
      else
         pos++;
   }

   append( buffer + pos );
}

void String::writeNumberHex( uint64 number, bool uppercase )
{
   // prepare the buffer
   char buffer[18];
   uint32 pos = 16;
   buffer[17] = '\0';

   byte base = uppercase ? 0x41 : 0x61;

   if ( number == 0 )
   {
      buffer[pos] = '0';
   }
   else {

      while( number != 0 ) {
         byte b = (byte)(number & 0xf);
         if ( b <= 9 )
            buffer[pos--] = (char) (b + 0x30);
         else {
            buffer[pos--] = (char) ( b - 10 + base );
         }

         number >>= 4;
      }
      pos++;
   }

   append( buffer + pos );
}

void String::writeNumberOctal( uint64 number )
{
   // prepare the buffer
   char buffer[32];
   uint32 pos = 30;
   buffer[31] = '\0';

   if ( number == 0 )
   {
      buffer[pos] = '0';
   }
   else {

      while( number != 0 ) {
         buffer[pos--] = (char) ((number & 0x7) + 0x30);
         number >>= 3;
      }
      pos++;
   }

   append( buffer + pos );
}

void String::writeNumber( int64 number, const String &format )
{
   char buffer[64];

   char bufFormat[32];
   if ( format.toCString( bufFormat, 32 ) == npos )
      return;

   sprintf( buffer, bufFormat, number );
   append( buffer );
}

void String::writeNumber( double number, const String &format )
{
   char buffer[64];

   char bufFormat[32];
   if ( format.toCString( bufFormat, 32 ) == npos )
      return;

   sprintf( buffer, bufFormat, number );
   append( buffer );
}


String &String::bufferize( const String &other )
{
   m_class->bufferize( this, &other );
   return *this;
}


String &String::bufferize()
{
   m_class->bufferize( this );
   return *this;
}


void String::trim( int mode )
{
   uint32 front = 0;
   uint32 len = length();

   // modes: 0 = all, 1 = front, 2 = back

   // first, trim from behind.
   if ( mode == 0 || mode == 2 ) {
      while( len > 0 )
      {
         uint32 chr = getCharAt( len - 1 );
         if( chr != ' ' && chr != '\n' && chr != '\r' && chr != '\t' )
         {
            break;
         }

         len --;
      }

      if ( len == 0 )
      {
         // string is actually empty.
         m_size = 0;
         return;
      }
   }

   // front trim
   if ( mode == 0 || mode == 1 ) {
      while( front < len )
      {
         uint32 chr = getCharAt( front );
         if( chr != ' ' && chr != '\n' && chr != '\r' && chr != '\t' )
         {
            break;
         }
         ++front;
      }
   }

   // front can't be == to len.
   if ( front > 0 )
   {
      // source and dest should be different, but it will work for this configuration.
      m_class->subString( this, front, len, this );
   }
   else {
      m_size = len * m_class->charSize();
   }
}

void String::lower()
{
   uint32 len = length();
   for( uint32 i = 0; i < len; i++ )
   {
      uint32 chr = getCharAt( i );
      if ( chr >= 'A' && chr <= 'Z' ) {
         setCharAt( i, chr | 0x20 );
      }
   }
}

void String::upper()
{
   uint32 len = length();
   for( uint32 i = 0; i < len; i++ )
   {
      uint32 chr = getCharAt( i );
      if ( chr >= 'a' && chr <= 'z' ) {
         setCharAt( i, chr & ~0x20 );
      }
   }
}

bool String::fromUTF8( const char *utf8 )
{
   // destroy old contents

   if ( m_allocated )
   {
      m_class->destroy( this );
      m_allocated = 0;
   }
   m_size = 0;

   // empty string?
   if ( *utf8 == 0 )
   {
      m_class = &csh::handler_static;
      return true;
   }

   // start scanning

   while ( *utf8 != 0 )
   {
      uint32 chr = 0;

      byte in = (byte) *utf8;

      // 4 bytes? -- pattern 1111 0xxx
      int count;
      if ( (in & 0xF8) == 0xF0 )
      {
         chr = (in & 0x7 ) << 18;
         count = 18;
      }
      // pattern 1110 xxxx
      else if ( (in & 0xF0) == 0xE0 )
      {
         chr = (in & 0xF) << 12;
         count = 12;
      }
      // pattern 110x xxxx
      else if ( (in & 0xE0) == 0xC0 )
      {
         chr = (in & 0x1F) << 6;
         count = 6;
      }
      else if( in < 0x80 )
      {
         chr = (uint32) in;
         count = 0;
      }
      // invalid pattern
      else {
         return false;
      }

      // read the other characters with pattern 0x10xx xxxx
      while( count > 0 )
      {
         count -= 6;

         utf8++;
         byte in = (byte) *utf8;

         if ( in == 0 ) {
            // short utf8 sequence
            return false;
         }
         else if( (in & 0xC0) != 0x80 )
         {
            // unrecognized pattern, protocol error
            return false;
         }
         chr |= (in & 0x3f) << count;
      }

      this->append( chr );

      utf8++;
   }

   return true;
}

bool String::startsWith( const String &str, bool icase ) const
{
   uint32 len = str.length();
   if ( len > length() ) return false;

   if ( icase )
   {
      for ( uint32 i = 0; i < len; i ++ )
      {
         uint32 chr1, chr2;
         if ( (chr1 = str.getCharAt(i)) != (chr2 = getCharAt(i)) )
         {
            if ( chr1 >= 'A' && chr1 <= 'z' && (chr1 | 0x20) != (chr2|0x20) )
               return false;
         }
      }
   }
   else
   {
      for ( uint32 i = 0; i < len; i ++ )
         if ( str.getCharAt(i) != getCharAt(i) )
            return false;
   }

   return true;
}


bool String::endsWith( const String &str, bool icase ) const
{
   uint32 len = str.length();
   uint32 mlen = length();
   uint32 start = mlen-len;

   if ( len > mlen ) return false;

   if ( icase )
   {
      for ( uint32 i = 0; i < len; ++i )
      {
         uint32 chr1, chr2;
         if ( (chr1 = str.getCharAt(i)) != (chr2 = getCharAt(i+start)) )
         {
            if ( chr1 >= 'A' && chr1 <= 'z' && (chr1 | 0x20) != (chr2|0x20) )
               return false;
         }
      }
   }
   else
   {
      for ( uint32 i = 0; i < len; ++i )
         if ( str.getCharAt(i) != getCharAt(i+start) )
            return false;
   }

   return true;
}

bool String::wildcardMatch( const String& wildcard, bool bIcase ) const
{
   const String* wcard = &wildcard;
   const String* cfr = this;

   uint32 wpos = 0, wlen = wcard->length();
   uint32 cpos = 0, clen = cfr->length();

   uint32 wstarpos = 0xFFFFFFFF;

   while ( cpos < clen )
   {
      if( wpos == wlen )
      {
         // we have failed the match; but if we had a star, we
         // may roll back to the starpos and try to match the
         // rest of the string
         if ( wstarpos != 0xFFFFFFFF )
         {
            wpos = wstarpos;
         }
         else {
            // no way, we're doomed.
            break;
         }
      }

      uint32 wchr = wcard->getCharAt( wpos );
      uint32 cchr = cfr->getCharAt( cpos );

      switch( wchr )
      {
         case '?': // match any character
            wpos++;
            cpos++;
         break;

         case '*':
         {
            // mark for restart in case of bad match.
            wstarpos = wpos;

            // match till the next character
            wpos++;
            // eat all * in a row
            while( wpos < wlen )
            {
               wchr = wcard->getCharAt( wpos );
               if ( wchr != '*' )
                  break;
               wpos++;
            }

            if ( wpos == wlen )
            {
               // we have consumed all the chars
               cpos = clen;
               break;
            }


            //eat up to next character
            while( cpos < clen )
            {
               cchr = cfr->getCharAt( cpos );
               if ( cchr == wchr )
                  break;
               cpos ++;
            }

            // we have eaten up the same char? --  then advance also wpos to prepare next loop
            if ( cchr == wchr )
            {
               wpos++;
               cpos++;
            }
            // else, everything must stay as it is, so cpos == clen but wpos != wlen causing fail.
         }
         break;

         default:
            if ( cchr == wchr ||
                  ( bIcase && cchr < 128 && wchr < 128 && (cchr | 32) == (wchr | 32) )
               )
            {
               cpos++;
               wpos++;
            }
            else
            {
               // can we retry?
               if ( wstarpos != 0xFFFFFFFF )
                  wpos = wstarpos;
               else {
                  // check failed -- we're doomed
                  return false;
               }
            }
      }
   }

   // at the end of the loop, the match is ok only if both the cpos and wpos are at the end
   return wpos == wlen && cpos == clen;
}

//============================================================
void string_deletor( void *data )
{
   delete (String *) data;
}

}

/* end of cstring.cpp */
