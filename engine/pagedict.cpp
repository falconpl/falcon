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

//=======================================================
// Iterator
//


PageDictIterator::PageDictIterator( PageDict *owner, const MapIterator &iter ):
   m_owner( owner ),
   m_iter( iter )
{
   m_versionNumber = owner->version();
}

bool PageDictIterator::next()
{
   return m_iter.next();
}

bool PageDictIterator::prev()
{
   return m_iter.prev();
}

FalconData *PageDictIterator::clone() const
{
   return new PageDictIterator( *this );
}

bool PageDictIterator::isValid() const
{
   if( m_versionNumber != m_owner->version() )
      return false;
   return m_iter.hasCurrent();
}

bool PageDictIterator::isOwner( void *collection ) const
{
   return m_owner == collection;
}

void PageDictIterator::invalidate()
{
   m_versionNumber--;
}

Item &PageDictIterator::getCurrent() const
{
   return *(Item *) m_iter.currentValue();
}

const Item &PageDictIterator::getCurrentKey() const
{
   return *(const Item *) m_iter.currentKey();
}

bool PageDictIterator::hasNext() const
{
   if( m_versionNumber != m_owner->version() )
      return false;
   return m_iter.hasNext();
}

bool PageDictIterator::hasPrev() const
{
   if( m_versionNumber != m_owner->version() )
      return false;
   return m_iter.hasPrev();
}

bool PageDictIterator::equal( const CoreIterator &other ) const
{
   if ( ! isValid() && ! other.isValid() )
      return true;

   if ( ! isValid() || ! other.isValid() )
      return false;

   if ( other.isOwner( m_owner ) )
   {
      return m_iter.equal( static_cast<const PageDictIterator *>(&other)->m_iter );
   }

   return false;
}

bool PageDictIterator::erase()
{
   if ( m_owner != 0 )
   {
      return m_owner->remove( *this );
   }

   return false;
}

bool PageDictIterator::insert( const Item & )
{
   return false;
}

//=======================================================
// Iterator
//

PageDict::PageDict( VMachine *vm ):
   CoreDict( vm, sizeof( this ) ),
   m_itemTraits( vm ),
   m_map( &m_itemTraits, &m_itemTraits ),
   m_version( 0 )
{}

PageDict::PageDict( VMachine *vm, uint32 pageSize ):
   CoreDict( vm, sizeof( PageDict ) ),
   m_itemTraits( vm ),
   m_map( &m_itemTraits, &m_itemTraits, (uint16) pageSize ),
   m_version( 0 )
{
}

PageDict::~PageDict()
{
}

uint32 PageDict::length() const
{
   return m_map.size();
}

DictIterator *PageDict::first()
{
   return new PageDictIterator( this, m_map.begin() );
}

DictIterator *PageDict::last()
{
   return new PageDictIterator( this, m_map.end() );
}


void PageDict::first( DictIterator &iter )
{
   PageDictIterator *ptr = static_cast<PageDictIterator *>( &iter );
   ptr->m_owner = this;
   ptr->m_iter = m_map.begin();
   ptr->m_versionNumber = version();
}

void PageDict::last( DictIterator &iter )
{
   PageDictIterator *ptr = static_cast<PageDictIterator *>( &iter );
   ptr->m_owner = this;
   ptr->m_iter = m_map.end();
   ptr->m_versionNumber = version();
}

Item *PageDict::find( const Item &key ) const
{
   return (Item *) m_map.find( &key );
}


bool PageDict::find( const Item &key, DictIterator &di )
{
   PageDictIterator *ptr = static_cast<PageDictIterator *>( &di );
   ptr->m_versionNumber = version();
   ptr->m_owner = this;

   return m_map.find( &key, ptr->m_iter );
}

DictIterator *PageDict::findIterator( const Item &key )
{
   MapIterator iter;
   if ( m_map.find( &key, iter ) )
   {
      return new PageDictIterator( this, iter );
   }
   return 0;
}

bool PageDict::remove( DictIterator &iter )
{
   if( ! iter.isOwner( this ) || ! iter.isValid() )
      return false;

   PageDictIterator *pit = static_cast< PageDictIterator *>( &iter );
   m_map.erase( pit->m_iter );

   m_version++;

   // maintain compatibility
   pit->m_versionNumber = m_version;
   return true;
}


bool PageDict::remove( const Item &key )
{
   if( m_map.erase( &key ) )
   {
      m_version++;
      return true;
   }

   return false;
}


void PageDict::insert( const Item &key, const Item &value )
{
   if( m_map.insert( &key, &value ) )
      m_version++;
}


void PageDict::smartInsert( DictIterator &iter, const Item &key, const Item &value )
{
   // todo
   insert( key, value );
}


bool PageDict::equal( const CoreDict &other ) const
{
   if ( &other == this )
      return true;
   return false;
}


void PageDict::merge( const CoreDict &dict )
{
   const_cast< CoreDict *>( &dict )->traverseBegin();

   Item key, value;
   while( const_cast< CoreDict *>( &dict )->traverseNext( key, value ) )
   {
      insert( key, value );
   }
}



CoreDict *PageDict::clone() const
{
   if ( m_map.size() == 0 )
      return new PageDict( origin() );

   PageDict *ret = new PageDict( origin(), m_map.order() );
   ret->merge( *this );
   return ret;
}

void PageDict::traverseBegin()
{
   m_traverseIter = m_map.begin();
}

bool PageDict::traverseNext( Item &key, Item &value )
{
   if( ! m_traverseIter.hasCurrent() )
      return false;

   key = *(Item *) m_traverseIter.currentKey();
   value = *(Item *) m_traverseIter.currentValue();
   m_traverseIter.next();
   return true;
}

void PageDict::clear()
{
   m_map.clear();
   m_version++;
}

}

/* end of dict.cpp */
