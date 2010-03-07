/*
   FALCON - The Falcon Programming Language.
   FILE: dict.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 21:55:38 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/pagedict.h>
#include <falcon/iterator.h>
#include <falcon/item.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/vm.h>

#if defined(__BORLANDC__)
   #include <string.h>
#else
   #include <cstring>
#endif

namespace Falcon
{

void PageDict::PageDictIterDeletor( Iterator* iter )
{
   MapIterator* mi = (MapIterator*) iter->data();
   delete mi;
}

//=======================================================
// Iterator
//

PageDict::PageDict():
   m_map( &m_itemTraits, &m_itemTraits ),
   m_mark( 0xFFFFFFFF )
{}

PageDict::PageDict( uint32 pageSize ):
   m_map( &m_itemTraits, &m_itemTraits, (uint16) pageSize ),
   m_mark( 0xFFFFFFFF )
{
}

PageDict::~PageDict()
{
}

uint32 PageDict::length() const
{
   return m_map.size();
}



Item *PageDict::find( const Item &key ) const
{
   return (Item *) m_map.find( &key );
}


bool PageDict::findIterator( const Item &key, Iterator &di )
{
   MapIterator *mi = (MapIterator *) di.data();
   bool ret = m_map.find( &key, *mi );
   di.data( mi );
   return ret;
}



bool PageDict::remove( const Item &key )
{
   if( m_map.erase( &key ) )
   {
      invalidateAllIters();
      return true;
   }

   return false;
}


void PageDict::put( const Item &key, const Item &value )
{
   if( m_map.insert( &key, &value ) )
      invalidateAllIters();
}


void PageDict::smartInsert( const Iterator &iter, const Item &key, const Item &value )
{
   // todo
   put( key, value );
}

void PageDict::merge( const ItemDict &dict )
{
   Iterator iter( const_cast<ItemDict*>(&dict) );

   while( iter.hasCurrent() )
   {
      Item& current = iter.getCurrent();
      put( iter.getCurrentKey(),
            current.isString() && current.asString()->isCore() ?
                  new CoreString( *current.asString() ) : current );
      iter.next();
   }
   invalidateAllIters();
}


PageDict *PageDict::clone() const
{
   PageDict *ret;

   if ( m_map.size() == 0 )
   {
      ret = new PageDict;
   }
   else
   {
      ret = new PageDict( m_map.order() );
      ret->merge( *this );
   }

   return ret;
}

const Item &PageDict::front() const
{
   if( m_map.empty() )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "PageDict::front" ) );

   return *(Item*) m_map.begin().currentValue();
}

const Item &PageDict::back() const
{
   if( m_map.empty() )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "PageDict::back" ) );

   return *(Item*) m_map.end().currentValue();
}

void PageDict::append( const Item& item )
{
   if( item.isArray() )
   {
      ItemArray& pair = item.asArray()->items();
      if ( pair.length() == 2 )
      {
         m_map.insert( &pair[0], &pair[1] );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime ).extra( "PageDict::append" ) );
   
   invalidateAllIters();

}

void PageDict::prepend( const Item& item )
{
   append( item );
   invalidateAllIters();
}

void PageDict::clear()
{
   m_map.clear();
   invalidateAllIters();
}

bool PageDict::empty() const
{
   return m_map.empty();
}

void PageDict::gcMark( uint32 gen )
{
   if( m_mark != gen )
   {
      m_mark = gen;
      Sequence::gcMark( gen );

      MapIterator iter = m_map.begin();
      while( iter.hasCurrent() )
      {
         memPool->markItem( *(Item*)iter.currentKey() );
         memPool->markItem( *(Item*)iter.currentValue() );
         iter.next();
      }
   }
}

//============================================================
// Iterator management.
//============================================================

void PageDict::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   
   MapIterator* mi = (MapIterator *) tgt.data();
   if ( mi == 0 )
   {
      mi = new MapIterator;
      tgt.data( mi );
      tgt.deletor( &PageDictIterDeletor );
   }

   *mi = tail ? m_map.end() : m_map.begin();
}


void PageDict::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   
   MapIterator* mi = (MapIterator *) tgt.data();
   if ( mi == 0 )
   {
      mi = new MapIterator;
      tgt.data( mi );
      tgt.deletor( &PageDictIterDeletor );
   }
      
   *mi = *(MapIterator *)source.data();
}


void PageDict::insert( Iterator &iter, const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PageDict::insert" ) );
}

void PageDict::erase( Iterator &iter )
{
   MapIterator* mi = (MapIterator*) iter.data();

   if ( ! mi->hasCurrent() )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
            .origin( e_orig_runtime ).extra( "PageDict::erase" ) );

   // map erase grants the validity of mi
   m_map.erase( *mi );
   invalidateAnyOtherIter( &iter );
}


bool PageDict::hasNext( const Iterator &iter ) const
{
   MapIterator* mi = (MapIterator*) iter.data();
   return mi->hasNext();
}


bool PageDict::hasPrev( const Iterator &iter ) const
{
   MapIterator* mi = (MapIterator*) iter.data();
   return mi->hasPrev();
}

bool PageDict::hasCurrent( const Iterator &iter ) const
{
   MapIterator* mi = (MapIterator*) iter.data();
   return mi->hasCurrent();
}


bool PageDict::next( Iterator &iter ) const
{
   MapIterator* mi = (MapIterator*) iter.data();
   return mi->next();
 }


bool PageDict::prev( Iterator &iter ) const
{
   MapIterator* mi = (MapIterator*) iter.data();
   return mi->prev();
}

Item& PageDict::getCurrent( const Iterator &iter )
{
   MapIterator* mi = (MapIterator*) iter.data();
   if ( mi->hasCurrent() )
      return *(Item*) mi->currentValue();

   throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "PageDict::getCurrent" ) );
}


Item& PageDict::getCurrentKey( const Iterator &iter )
{
   MapIterator* mi = (MapIterator*) iter.data();
   if ( mi->hasCurrent() )
         return *(Item*) mi->currentKey();

   throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "PageDict::getCurrent" ) );

}


bool PageDict::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.position() == second.position();
}


}

/* end of dict.cpp */
