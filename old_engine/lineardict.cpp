/*
   FALCON - The Falcon Programming Language.
   FILE: lineardict.cpp

   Linear dictionary
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 21:55:38 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/lineardict.h>
#include <falcon/iterator.h>
#include <falcon/item.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/vm.h>
#include <string.h>
#include <cstring>

namespace Falcon
{

LinearDict::LinearDict():
   m_size(0),
   m_alloc(0),
   m_data(0),
   m_mark( 0xFFFFFFFF )
{}

LinearDict::LinearDict( uint32 size ):
   m_mark( 0xFFFFFFFF )
{
   m_data = (LinearDictEntry *) memAlloc( esize( size ) );
   length(0);
   allocated( size );
}

LinearDict::~LinearDict()
{
   if ( m_data != 0 )
      memFree( m_data );
}

uint32 LinearDict::length() const
{
   return m_size;
}

bool LinearDict::empty() const
{
   return m_size == 0;
}

const Item &LinearDict::front() const
{
   if( m_size == 0 )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "LinearDict::front" ) );

   return m_data[0].value();
}

const Item &LinearDict::back() const
{
   if( m_size == 0 )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "LinearDict::back" ) );

   return m_data[m_size-1].value();
}

void LinearDict::append( const Item& item )
{
   if( item.isArray() )
   {
      ItemArray& pair = item.asArray()->items();
      if ( pair.length() == 2 )
      {
         put( pair[0], pair[1] );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime ).extra( "LinearDict::append" ) );

}

void LinearDict::prepend( const Item& item )
{
   append( item );
   invalidateAllIters();
}



Item *LinearDict::find( const Item &key ) const
{
   uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      return &m_data[ posHint ].value();
   }

   return 0;
}

bool LinearDict::findIterator( const Item &key, Iterator &iter )
{
   uint32 posHint;

   // Insert supports substitution semantics.
   bool val = findInternal( key, posHint );
   iter.position( posHint );
   return val;
}

bool LinearDict::remove( const Item &key )
{
  uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      removeAt( posHint );
      m_invalidPos = posHint;
      invalidateIteratorOnCriterion();
      return true;
   }

   return false;
}

void LinearDict::put( const Item &key, const Item &value )
{
   uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      m_data[ posHint ].value( value );
      return;
   }

   // Entry not found, must be added
   addInternal( posHint, key, value );
   m_invalidPos = posHint;
   invalidateIteratorOnCriterion();
}

void LinearDict::smartInsert( const Iterator &iter, const Item &key, const Item &value )
{
   if ( m_size == 0 )
   {
      addInternal( 0, key, value );
      return;
   }

   if ( iter.hasCurrent() )
   {
      uint32 posHint = (uint32) iter.position();

      // right position?
      if (  key == m_data[posHint].key() )
      {
         m_data[ posHint ].value( value );
         return;
      }

      // not right, but good for insertion?
      if (
         ( posHint == 0 || key > m_data[posHint-1].key() ) &&
         ( posHint == m_size || key < m_data[posHint].key() ) )
      {
         addInternal( posHint, key, value );
         m_invalidPos = posHint;
         invalidateIteratorOnCriterion();
         return;
      }
   }

   // nothing to do, perform a full search
   put( key, value );
}


void LinearDict::merge( const ItemDict &dict )
{
   if ( dict.length() > 0 )
   {
      m_alloc = m_size + dict.length();

      m_data = (LinearDictEntry*) memRealloc( m_data, sizeof( LinearDictEntry )*m_alloc );
      Iterator iter( const_cast<ItemDict*>( &dict ) );

      while( iter.hasCurrent() )
      {
         put( iter.getCurrentKey(), iter.getCurrent() );
         iter.next();
      }
   }
   
   invalidateAllIters();
}  


bool LinearDict::addInternal( uint32 pos, const Item &key, const Item &value )
{
   if ( pos > m_size )
      return false;

   // haven't we got enough space?
   if ( m_alloc <= m_size  )
   {
      m_alloc = m_size + flc_DICT_GROWTH;
      LinearDictEntry *mem = (LinearDictEntry *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem, m_data, esize( pos ) );

      LinearDictEntry *entry = (LinearDictEntry *) (mem + pos);
      entry->key( key );
      entry->value( value );
      if ( pos < m_size )
         memcpy( mem+pos + 1, m_data + pos, esize( m_size - pos ) );

      if ( m_data != 0 )
         memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < m_size )
         memmove( m_data + pos + 1, m_data + pos,  esize( m_size - pos ) );

      LinearDictEntry *entry = (LinearDictEntry *) (m_data + pos);
      entry->key( key );
      entry->value( value );
  }

   length( m_size + 1 );
   return true;
}


bool LinearDict::removeAt( uint32 pos )
{
   if ( pos >= m_size )
      return false;

   if ( pos < m_size - 1 )
      memmove( m_data + pos, m_data + pos + 1, esize( m_size - pos ) );
   // otherwise, there's nothing to move...

   length( m_size - 1 );

   // for now, do not reallocate.
   m_invalidPos = pos;
   invalidateIteratorOnCriterion();

   return true;
}


bool LinearDict::findInternal( const Item &key, uint32 &ret_pos ) const
{
   uint32 lower = 0, higher, point;
   higher = m_size;

   if ( higher == 0 ) {
      ret_pos = 0;
      return false;
   }
   higher --;

   point = higher / 2;
   LinearDictEntry *current;
   int comparation;

   while ( true )
   {
      // get the table row
      current = m_data + point;

      comparation = key.compare( current->key() );

      if ( comparation == 0 )
      {
         ret_pos = point;
         return true;
      }
      else
      {
         if ( lower == higher )  // not found
         {
            break;
         }
         // last try. In pair sized dictionaries, it can be also in the other node
         else if ( lower == higher -1 )
         {
            // key is EVEN less than the lower one
            if ( comparation < 0  )
            {
               ret_pos = lower;
               return false;
            }

            // being integer math, ulPoint is rounded by defect and has
            // already looked at the ulLower position
            point = lower = higher;
            // try again
            continue;
         }

         if ( comparation > 0 )
         {
            lower = point;
         }
         else
         {
            higher = point;
         }
         point = ( lower + higher ) / 2;
      }
   }

   // entry not found, but signal the best match anyway
   ret_pos =  comparation > 0 ? higher + 1 : higher;

   return false;
}


LinearDict *LinearDict::clone() const
{
   LinearDict *ret;

   if ( m_size == 0 )
   {
      ret = new LinearDict();
   }
   else
   {
      ret = new LinearDict( m_size );
      ret->length( m_size );
      memcpy( ret->m_data, m_data, esize( m_size ) );

      // duplicate strings
      for ( uint32 i = 0; i < m_size; ++i )
      {
         Item& item = m_data[i].m_value;

         if( item.isString() && item.asString()->isCore() )
         {
            item = new CoreString( *item.asString() );
         }
      }
   }

   return ret;
}


void LinearDict::clear()
{
   memFree( m_data );
   m_data = 0;
   m_alloc = 0;
   m_size = 0;
   invalidateAllIters();
}

void LinearDict::gcMark( uint32 gen )
{
   if ( m_mark  != gen )
   {
      m_mark = gen;

      Sequence::gcMark( gen );

      for( uint32 i = 0; i < length(); ++i )
      {
         memPool->markItem( m_data[i].key() );
         memPool->markItem( m_data[i].value() );
      }
   }
}

//============================================================
// Iterator management.
//============================================================

void LinearDict::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   tgt.position( tail ? (length()>0? length()-1: 0) : 0 );
}


void LinearDict::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   tgt.position( source.position() );
}

void LinearDict::insert( Iterator &iter, const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "LinearDict::insert" ) );
}

void LinearDict::erase( Iterator &iter )
{
   if ( iter.position() >= length() )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
            .origin( e_orig_runtime ).extra( "LinearDict::erase" ) );
   
   uint32 pos = (uint32)iter.position();
   if ( pos < m_size - 1 )
      memmove( m_data + pos, m_data + pos + 1, esize( m_size - pos ) );
   // otherwise, there's nothing to move...

   length( m_size - 1 );

   // the next item has automatically moved on the position pointed by us
   // but the other iterators should be killed.
   invalidateAnyOtherIter( &iter );
}


bool LinearDict::hasNext( const Iterator &iter ) const
{
   return iter.position()+1 < length();
}


bool LinearDict::hasPrev( const Iterator &iter ) const
{
   return iter.position() > 0;
}

bool LinearDict::hasCurrent( const Iterator &iter ) const
{
   return iter.position() < length();
}


bool LinearDict::next( Iterator &iter ) const
{
   if ( iter.position() < length() )
   {
      iter.position( iter.position() + 1 );
      return iter.position() < length();
   }

   return false;
}


bool LinearDict::prev( Iterator &iter ) const
{
   if ( iter.position() > 0 )
   {
      iter.position( iter.position() - 1 );
      return true;
   }

   iter.position( length() );
   return false;
}

Item& LinearDict::getCurrent( const Iterator &iter )
{
   if ( iter.position() < length() )
      return m_data[ iter.position() ].value();

   throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "LinearDict::getCurrent" ) );
}


Item& LinearDict::getCurrentKey( const Iterator &iter )
{
   if ( iter.position() < length() )
         return m_data[ iter.position() ].key();

   throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "LinearDict::getCurrent" ) );

}


bool LinearDict::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.position() == second.position();
}

bool LinearDict::onCriterion( Iterator* elem ) const
{
   return elem->position() < m_invalidPos;
}

}

/* end of dict.cpp */
