/*
   FALCON - The Falcon Programming Language.
   FILE: corearray.cpp

   Language level array implementation
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
#include <falcon/string.h>
#include <falcon/vm.h>
#include <string.h>
#include <falcon/lineardict.h>
#include <falcon/coretable.h>


namespace Falcon {

CoreArray::CoreArray():
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{
   m_itemarray.owner( this );
}


CoreArray::CoreArray( const CoreArray& other ):
   m_itemarray( other.m_itemarray ),
   m_table( other.m_table ),
   m_tablePos( other.m_tablePos )
{
   m_itemarray.owner( this );

   if ( other.m_bindings != 0 )
   {
      m_bindings = static_cast<CoreDict*>( other.m_bindings->clone() );
      m_bindings->gcMark( mark() );
   }
   else
      m_bindings = 0;
}

CoreArray::CoreArray( uint32 prealloc ):
   m_itemarray( prealloc ),
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{
   m_itemarray.owner( this );
}

CoreArray::CoreArray( Item *buffer, uint32 size, uint32 alloc ):
   m_itemarray( buffer, size, alloc ),
   m_bindings(0),
   m_table(0),
   m_tablePos(0)
{
   m_itemarray.owner( this );
}

CoreArray::~CoreArray()
{
}

const String& CoreArray::name() const {
   static String name( "Array" );
   return name;
}

void CoreArray::readyFrame( VMachine* vm, uint32 paramCount )
{
   vm->prepareFrame( this, paramCount );
}


CoreDict *CoreArray::makeBindings()
{
   if ( m_bindings == 0 )
   {
      m_bindings = new CoreDict( new LinearDict( ) );
      m_bindings->put( new CoreString( "self" ), this );
      m_bindings->gcMark( mark() );
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

   m_bindings->put( new CoreString( name ), ref );
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

   item = *p->dereference();
   item.methodize( this );  // may fail but it's ok
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
               target = m_itemarray[length()+pos];
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               target = m_itemarray[pos];
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
               target = m_itemarray[length()+pos];
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               target = m_itemarray[pos];
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


CoreArray* CoreArray::partition( int32 start, int32 end ) const
{
   int32 size;
   Item *buffer;

   if ( start < 0 )
      start = m_itemarray.m_size + start;
   if ( start < 0 || start >= (int32) m_itemarray.m_size )
      return 0;

   if ( end < 0 )
      end = m_itemarray.m_size + end;
   if ( end < 0 || end > (int32) m_itemarray.m_size )
      return 0;

   if( end < start ) {
      size = start - end + 1;
      buffer = (Item *) memAlloc( m_itemarray.esize( size ) );

      for( int i = 0; i < size; i ++ )
         buffer[i] = m_itemarray.m_data[start - i];
   }
   else {
      if( end == start ) {
         return new CoreArray;
      }
      size = end - start;
      buffer = (Item *) memAlloc( m_itemarray.esize( size ) );
      memcpy( buffer, m_itemarray.m_data + start, m_itemarray.esize( size )  );
   }

   return new CoreArray( buffer, size, size );
}

CoreArray *CoreArray::clone() const
{
   return new CoreArray( *this );
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
                  m_itemarray[length()+pos] = new CoreString( *target.asString() );
               else
                  m_itemarray[length()+pos] = target;
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
               if ( target.isString() )
                  m_itemarray[pos] = new CoreString( *target.asString() );
               else
                  m_itemarray[pos] = target;
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
                  m_itemarray[length()+pos] = new CoreString( *target.asString() );
               else
                  m_itemarray[length()+pos] = target;
               return;
            }
         }
         else
         {
            if( pos < (int) length() )
            {
                if ( target.isString() )
                  m_itemarray[pos] = new CoreString( *target.asString() );
               else
                  m_itemarray[pos] = target;
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

void CoreArray::gcMark( uint32 gen )
{
   CoreArray *array = this;

   if( array->mark() != gen )
   {
      array->mark(gen);
      array->items().gcMark(gen);

      // mark also the bindings
      if ( array->bindings() != 0 )
      {
         array->bindings()->gcMark( gen );
      }

      // and also the table
      if ( array->table() != 0 && array->table()->mark() != gen )
      {
         array->table()->gcMark( gen );
      }
   }
}

}

/* end of corearray.cpp */
