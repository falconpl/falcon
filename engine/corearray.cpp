/*
   FALCON - The Falcon Programming Language.
   FILE: corearray.cpp

   Core arrays are meant to hold item pointers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of core arrays.
   Core arrays are meant to hold item pointers.
*/

#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/vm.h>
#include <string.h>
#include <falcon/lineardict.h>
#include <falcon/coretable.h>

namespace Falcon {

CoreArray::CoreArray( ):
   Garbageable(),
   m_alloc(0),
   m_size(0),
   m_data(0),
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{}


CoreArray::CoreArray( uint32 prealloc ):
   Garbageable(),
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{
   m_data = (Item *) memAlloc( esize(prealloc) );
   m_alloc = prealloc;
   m_size = 0;
}

CoreArray::CoreArray( Item *buffer, uint32 size, uint32 alloc ):
   Garbageable(),
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{
   m_data = buffer;
   m_alloc = alloc;
   m_size = size;
}

CoreArray::~CoreArray()
{
   if ( m_data != 0 )
      memFree( m_data );
}


void CoreArray::append( const Item &ndata )
{
   // create enough space to hold the data
   if ( m_alloc <= m_size )
   {
      m_alloc = m_size + flc_ARRAY_GROWTH;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
   }
   m_data[ m_size ] = ndata;
   m_size++;
}

void CoreArray::merge( const CoreArray &other )
{
   if ( m_table != 0 )
      return;

   if ( other.m_size == 0 )
      return;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
   }

   memcpy( m_data + m_size, other.m_data, esize( other.m_size ) );
   m_size += other.m_size;
}

void CoreArray::prepend( const Item &ndata )
{
   if ( m_table != 0 )
      return;

   // create enough space to hold the data
   Item *mem = (Item *) memAlloc( esize(m_size + 1) );
   m_alloc = m_size + 1;
   if ( m_size != 0 )
      memcpy( mem + 1, m_data, esize( m_size ) );
   mem[0] = ndata;
   if ( m_size != 0 )
      memFree( m_data );
   m_data = mem;
   m_size++;
}

void CoreArray::merge_front( const CoreArray &other )
{
   if ( m_table != 0 )
      return;

   if ( other.m_size == 0 )
      return;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      memcpy( mem , other.m_data, esize( other.m_size ) );
      if ( m_size > 0 ) {
         memcpy( mem + other.m_size, m_data, esize( m_size ) );
         memFree( m_data );
      }

      m_size = m_alloc;
      m_data = mem;
   }
   else {
      memmove( m_data + other.m_size, m_data, esize( m_size ) );
      memcpy( m_data, other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }
}

bool CoreArray::insert( const Item &ndata, int32 pos )
{
   if ( m_table != 0 )
      return false;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > (int32) m_size )
      return false;

   if ( m_alloc <= m_size ) {
      m_alloc = m_size + flc_ARRAY_GROWTH;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );
      if ( pos < (int32)m_size )
         memcpy( mem + pos + 1, m_data + pos , esize(m_size - pos) );

      mem[ pos ] = ndata;
      m_size++;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + pos + 1, m_data+pos, esize( m_size - pos) );
      m_data[pos] = ndata;
      m_size ++;
   }
   return true;
}

bool CoreArray::insert( const CoreArray &other, int32 pos )
{
   if ( m_table != 0 )
      return false;

   if ( other.m_size == 0 )
      return true;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > (int32)m_size )
      return false;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < (int32)m_size )
         memcpy( mem + pos + other.m_size, m_data + pos , esize(m_size - pos) );

      memcpy( mem + pos , other.m_data, esize( other.m_size ) );

      m_size = m_alloc;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + other.m_size + pos, m_data + pos, esize(m_size - pos ) );
      memcpy( m_data + pos , other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }

   return true;
}

bool CoreArray::remove( int32 first, int32 last )
{
   if ( m_table != 0 )
      return false;

   if ( first < 0 )
      first = m_size + first;
   if ( first < 0 || first >= (int32)m_size )
      return false;

   if ( last < 0 )
      last = m_size + last;
   if ( last < 0 || last > (int32)m_size )
      return false;

   if( first > last ) {
      int32 temp = first;
      first = last;
      // last can't be < first if it was == size.
      last = temp+1;
   }

   uint32 rsize = last - first;
   if ( last < (int32)m_size )
      memmove( m_data + first, m_data + last, esize(m_size - last) );
   m_size -= rsize;
   return true;
}

int32 CoreArray::find( const Item &itm ) const
{
   for( uint32 i = 0; i < m_size; i ++ )
   {
      if ( itm == m_data[ i ] )
         return (int32) i;
   }

   return -1;
}


bool CoreArray::remove( int32 first )
{
   if ( m_table != 0 )
      return false;

   if ( first < 0 )
      first = m_size + first;
   if ( first < 0 || first >= (int32)m_size )
      return false;

   if ( first < (int32)m_size - 1 )
      memmove( m_data + first, m_data + first + 1, esize(m_size - first) );
   m_size --;
   return true;
}

bool CoreArray::change( const CoreArray &other, int32 begin, int32 end )
{
   if ( begin < 0 )
      begin = m_size + begin;
   if ( begin < 0 || begin > (int32)m_size )
      return false;

   if ( end < 0 )
      end = m_size + end;
   if ( end < 0 || end > (int32)m_size )
      return false;

   if( begin > end ) {
      int32 temp = begin;
      begin = end;
      end = temp+1;
   }

   int32 rsize = end - begin;

   // we're considering end as "included" from now on.
   // this considers also negative range which already includes their extreme.
   if ( m_size - rsize + other.m_size > m_alloc )
   {
      m_alloc =  m_size - rsize +other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( begin > 0 )
         memcpy( mem, m_data, esize( begin ) );
      if ( other.m_size > 0 )
         memcpy( mem + begin, other.m_data, esize( other.m_size ) );
      if ( end < (int32) m_size )
         memcpy( mem + begin + other.m_size, m_data + end, esize(m_size - end) );

      memFree( m_data );
      m_data = mem;
      m_size = m_alloc;
   }
   else {
      if ( end < (int32)m_size )
         memmove( m_data + begin + other.m_size, m_data + end, esize(m_size - end) );

      if ( other.m_size > 0 )
         memcpy( m_data + begin, other.m_data, esize( other.m_size ) );
      m_size = m_size - rsize + other.m_size;
   }

   return true;
}

bool CoreArray::insertSpace( uint32 pos, uint32 size )
{
   if ( m_table != 0 )
      return false;

   if ( size == 0 )
      return true;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > m_size )
      return false;

   if ( m_alloc < m_size + size ) {
      m_alloc = m_size + size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < m_size )
         memcpy( mem + pos + size, m_data + pos , esize(m_size - pos) );

      for( uint32 i = pos; i < pos + size; i ++ )
         m_data[i] = Item();

      m_size += size;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < m_size )
      memmove( m_data + size + pos, m_data + pos, esize(m_size - pos) );
      for( uint32 i = pos; i < pos + size; i ++ )
         m_data[i] = Item();
      m_size += size;
   }

   return true;
}


CoreArray *CoreArray::partition( int32 start, int32 end ) const
{
   int32 size;
   Item *buffer;

   if ( start < 0 )
      start = m_size + start;
   if ( start < 0 || start >= (int32)m_size )
      return 0;

   if ( end < 0 )
      end = m_size + end;
   if ( end < 0 || end > (int32)m_size )
      return 0;

   if( end < start ) {
      size = start - end + 1;
      buffer = (Item *) memAlloc( esize( size ) );

      for( int i = 0; i < size; i ++ )
         buffer[i] = m_data[start - i];
   }
   else {
      if( end == start ) {
         return new CoreArray();
      }
      size = end - start;
      buffer = (Item *) memAlloc( esize( size ) );
      memcpy( buffer, m_data + start, esize( size )  );
   }
   return new CoreArray( buffer, size, size );
}

CoreArray *CoreArray::clone() const
{
   Item *buffer = (Item *) memAlloc( esize( m_size ) );
   memcpy( buffer, m_data, esize( m_size )  );
   CoreArray *ca = new CoreArray( buffer, m_size, m_size );
   ca->m_table = m_table;
   ca->m_tablePos = m_tablePos;

   if ( m_bindings != 0 )
   {
      ca->m_bindings = m_bindings->clone();
      ca->m_bindings->mark( mark() );
   }

   return ca;
}


void CoreArray::resize( uint32 size ) {
   if ( size == 0 ) {
      if ( m_data != 0 ) {
         memFree( m_data );
         m_data = 0;
      }
   }
   // use this request also to force size in shape with alloc.
   else if ( m_size != size || m_alloc != m_size ) {
      m_data = (Item *) memRealloc( m_data, esize( size ) );
      for( uint32 i = m_size; i < size; i++ ) {
         m_data[ i ].type( FLC_ITEM_NIL );
      }
   }
   m_size = size;
   m_alloc = size;
}

void CoreArray::reserve( uint32 size ) {
   if ( size > m_alloc ) {
      m_data = (Item *) memRealloc( m_data, esize( size ) );
      m_alloc = size;
   }
}

CoreDict *CoreArray::makeBindings()
{
   if ( m_bindings == 0 )
   {
      m_bindings = new LinearDict( );
      m_bindings->insert( new CoreString( "self" ), this );
      m_bindings->mark( mark() );
   }

   return m_bindings;
}

Item* CoreArray::getProperty( const String &name )
{
   Item *found = 0;

   if ( m_bindings != 0 )
   {
      found = m_bindings->find( name );
      if ( found )
         return found->dereference();
   }

   // we didn't find it.
   if ( m_table )
   {
      CoreTable *table = reinterpret_cast<CoreTable *>( m_table->getFalconData() );
      uint32 pos = table->getHeaderPos( name );
      if ( pos != CoreTable::noitem )
      {
         found = (*this)[pos].dereference();
         if ( found->isNil() && ! found->isOob() )
            found = table->getHeaderData( pos )->dereference();

      }
   }

   return found;
}

void CoreArray::setProperty( const String &name, const Item &data )
{
   if ( m_bindings != 0 )
   {
      Item* found = m_bindings->find( name );
      if ( found != 0 ) {
         *found = data;
         return;
      }
   }

   if ( m_table != 0 )
   {
      CoreTable *table = reinterpret_cast<CoreTable *>( m_table->getFalconData() );
      uint32 pos = table->getHeaderPos( name );
      if ( pos != CoreTable::noitem )
      {
         *(*this)[pos].dereference() = data;
         return;
      }
   }

   m_bindings = makeBindings();
   Item ref;
   VMachine* vm = VMachine::getCurrent();
   if ( vm )
      vm->referenceItem( ref, *const_cast<Item*>(&data) );

   m_bindings->insert( new CoreString( name ), ref );
}


void CoreArray::readProperty( const String &prop, Item &item )
{
   Item *p = getProperty( prop );
   if ( p == 0 )
   {
      // try to find a generic method
      VMachine* vm = VMachine::getCurrent();
      fassert( vm != 0);
      CoreClass* cc = vm->getMetaClass( FLC_ITEM_ARRAY );
      uint32 id;
      if ( cc == 0 || ! cc->properties().findKey( prop, id ) )
      {
         throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
      }

      p = cc->properties().getValue( id );
      fassert( ! p->isReference() );
   }

   if ( p->isFunction() )
      item.setMethod( this, p->asFunction() );
   else
      item = *p;
}

void CoreArray::writeProperty( const String &prop, const Item &item )
{
   setProperty( prop, *item.dereference() );
}

void CoreArray::readIndex( const Item &index, Item &target )
{
   switch ( index.type() )
   {
      case FLC_ITEM_INT:
      {
         register int32 pos = (int32) index.asInteger();
         if ( pos < 0 )
         {
            if ( -pos <= (int32) length() )
            {
               target = elements()[length()+pos];
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               target = elements()[pos];
               return;
            }
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         register int32 pos = (int32) index.asNumeric();
         if ( pos < 0 )
         {
            if ( -pos <= (int32) length() )
            {
               target = elements()[length()+pos];
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               target = elements()[pos];
               return;
            }
         }
      }
      break;

      case FLC_ITEM_RANGE:
      {
         // open ranges?
         if ( index.asRangeIsOpen() &&
              ((index.asRangeStart() >= 0 && (int) length() <= index.asRangeStart() ) ||
              (index.asRangeStart() < 0 && (int) length() < -index.asRangeStart() ))
              )
         {
            target = new CoreArray();
            return;
         }

         register int32 end = (int32)(index.asRangeIsOpen() ? length() : index.asRangeEnd());
         CoreArray* array = partition( (int32) index.asRangeStart(), end );
         if ( array != 0 )
         {
            target = array;
            return;
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         readIndex( index.asReference()->origin(), target );
         return;
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "LDP" ) );
}

void CoreArray::writeIndex( const Item &index, const Item &target )
{
  switch ( index.type() )
   {
      case FLC_ITEM_INT:
      {
         register int32 pos = (int32) index.asInteger();
         if ( pos < 0 )
         {
            if ( -pos <= (int32) length() )
            {
               if ( target.isString() )
                  elements()[length()+pos] = new CoreString( *target.asString() );
               else
                  elements()[length()+pos] = target;
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               if ( target.isString() )
                  elements()[pos] = new CoreString( *target.asString() );
               else
                  elements()[pos] = target;
               return;
            }
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         register int32 pos = (int32) index.asNumeric();
         if ( pos < 0 )
         {
            if ( -pos <= (int32) length() )
            {
               if ( target.isString() )
                  elements()[length()+pos] = new CoreString( *target.asString() );
               else
                  elements()[length()+pos] = target;
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
                if ( target.isString() )
                  elements()[pos] = new CoreString( *target.asString() );
               else
                  elements()[pos] = target;
               return;
            }
         }
      }
      break;

      case FLC_ITEM_RANGE:
      {
         int32 end = (int32)(index.asRangeIsOpen() ? length() : index.asRangeEnd());
         int32 start = (int32) index.asRangeStart();
         const Item *tgt = target.dereference();

         if( tgt->isArray() )
         {
            if( change( *target.asArray(), (int32)start, (int32)end ) )
               return;
         }
         else
         {
            // if it's not a plain insert...
            if ( start != end )
            {
               // before it's too late.
               if ( start <  0 )
                  start = length() + start;

               if ( end <  0 )
                  end = length() + end;

               if( ! remove( start, end ) )
                  throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "STP" ) );

               if ( start > end )
                  start = end;
            }

            if ( tgt->isString() )
            {
                if ( insert( new CoreString( *tgt->asString() ), start ) )
                  return;
            }
            else {
               if( insert( *tgt, start ) )
                  return;
            }
         }

      }
      break;

      case FLC_ITEM_REFERENCE:
         writeIndex( index.asReference()->origin(), target );
         return;
   }


   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "STP" ) );
}


}

/* end of corearray.cpp */
