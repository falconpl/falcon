/*
   FALCON - The Falcon Programming Language.
   FILE: itemlist.cpp

   List of Falcon Items
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   List of Falcon Items implementation
*/

#include <falcon/setup.h>
#include <falcon/itemlist.h>
#include <falcon/mempool.h>

namespace Falcon {

ItemList::ItemList( const ItemList &l ):
   m_iters(0)
{
   ItemListElement *h = l.m_head;
   if ( h == 0 )
   {
      m_head = 0;
      m_tail = 0;
      m_size = 0;
      return;
   }

   m_head = new ItemListElement( h->item(), 0, 0 );
   ItemListElement *h1 = m_head;
   h = h->next();
   while( h != 0 )
   {
      h1->next( new ItemListElement( h->item(), h1, 0 ) );
      h1 = h1->next();
      h = h->next();
   }
   m_tail = h1;

   m_size = l.m_size;
}



UserData *ItemList::clone() const
{
   return new ItemList( *this );
}


const Item &ItemList::front() const
{
   return m_head->item();
}


const Item &ItemList::back() const
{
   return m_tail->item();
}


ItemListElement *ItemList::first() const
{
   return m_head;
}


ItemListElement *ItemList::last() const
{
   return m_tail;
}


ItemListIterator *ItemList::getIterator( bool tail )
{
   return new ItemListIterator( this, tail ? m_tail : m_head );
}


void ItemList::push_back( const Item &itm )
{
   if ( m_tail == 0 )
   {
      m_head = m_tail = new ItemListElement( itm );
      m_size = 1;
   }
   else {
      m_tail->next( new ItemListElement( itm, m_tail, 0 ) );
      m_tail = m_tail->next();
      m_size++;
   }
}


void ItemList::pop_back()
{
   if( m_tail != 0 )
   {
      m_tail = m_tail->prev();

      // was the only element?
      if ( m_tail == 0 )
      {
         // delete it and update head.
         delete m_head;
         m_head = 0;
      }
      else {
         delete m_tail->next();
         m_tail->next(0);
      }
      m_size--;
   }
}


void ItemList::push_front( const Item &itm )
{
   if ( m_head == 0 )
   {
      m_head = m_tail = new ItemListElement( itm );
      m_size = 1;
   }
   else {
      m_head->prev( new ItemListElement( itm, 0, m_head ) );
      m_head = m_head->prev();
      m_size++;
   }
}


void ItemList::pop_front()
{
   if( m_head != 0 )
   {
      m_head = m_head->next();

      // was the only element?
      if ( m_head == 0 )
      {
         // delete it and update head.
         delete m_tail;
         m_tail = 0;
      }
      else {
         delete m_head->prev();
         m_head->prev(0);
      }
      m_size--;
   }
}


bool ItemList::erase( CoreIterator *iter )
{
   if ( iter->isOwner( this ) )
   {
      ItemListIterator *li = static_cast<ItemListIterator *>( iter );
      ItemListElement *elem = li->getCurrentElement();
      if ( elem != 0 )
      {
         elem = li->getCurrentElement();
         li->setCurrentElement( elem->next() );
         erase( elem );
         return true;
      }
   }

   return false;
}


ItemListElement *ItemList::erase( ItemListElement *elem )
{
   if ( m_head == 0 )
   {
      //?
      return 0;
   }

   if ( elem == m_head )
   {
      m_head = m_head->next();
      // was the last element?
      if ( m_head == 0 )
         m_tail = 0;
      else
         m_head->prev( 0 );
   }
   else if ( elem == m_tail )
   {
      // cannot be also  == head, so the list is not empty
      m_tail = m_tail->prev();
      m_tail->next(0);
   }
   else
   {
      // we know we have a valid prev and next
      elem->prev()->next( elem->next() );
      elem->next()->prev( elem->prev() );
   }

   ItemListElement *retval = elem->next();
   notifyDeletion( elem );
   delete elem;
   m_size--;
   return retval;
}

void ItemList::insert( ItemListElement *elem, const Item &item )
{
   if ( elem == 0 )
   {
      push_back( item );
      return;
   }

   // we have a valid element.
   // it may be head, tail or both.
   if ( elem == m_head )
   {
      m_head->prev( new ItemListElement( item, 0, m_head ) );
      m_head = m_head->prev();
   }
   // but we don't have to move the tail which stays where it is.
   else
   {
      ItemListElement *en = new ItemListElement( item, elem->prev(), elem );
      elem->prev()->next( en );
      elem->prev( en );
   }

   m_size++;
}

bool ItemList::insert( CoreIterator *iter, const Item &item )
{
   if ( iter->isOwner( this ) )
   {
      ItemListIterator *li = static_cast< ItemListIterator *>( iter );
      ItemListElement *elem = li->getCurrentElement();
      insert( elem, item );
      // was a tail insertion?
      li->setCurrentElement( elem == 0 ? m_tail : elem->prev() );
      return true;
   }

   return false;
}

void ItemList::clear()
{
   ItemListElement *h = m_head;
   while( h != 0 )
   {
      ItemListElement *nx = h->next();
      delete h;
      h = nx;
   }
   m_head = 0;
   m_tail = 0;
   m_size = 0;
}


void ItemList::addIterator( ItemListIterator *iter )
{
   // add the iterator on top
   iter->m_next = m_iters;
   iter->m_prev = 0;
   if ( m_iters != 0 )
      m_iters->m_prev = iter;
   m_iters = iter;
}

void ItemList::removeIterator( ItemListIterator *iter )
{
   if ( iter->m_prev != 0 )
   {
      iter->m_prev->m_next = iter->m_next;
   }
   else {
      m_iters = iter->m_next;
   }

   if ( iter->m_next != 0 )
   {
      iter->m_next->m_prev = iter->m_prev;
      iter->m_next = 0;
   }
   iter->m_prev = 0;

}

void ItemList::notifyDeletion( ItemListElement *elem )
{
   ItemListIterator *iter = m_iters;
   while( iter != 0 )
   {
      ItemListIterator *in = iter->m_next;
      // invalidate would disrupt the list
      if ( elem == iter->m_element )
         iter->invalidate();
      iter = in;
   }
}

void ItemList::gcMark( MemPool *mp )
{
   // we don't have to record the mark byte, as we woudln't have been called
   // if the coreobject holding us had the right mark.

   ItemListElement *h = m_head;
   while( h != 0 )
   {
      mp->markItem( h->item() );
      h = h->next();
   }
}



//====================================================

ItemListIterator::ItemListIterator( ItemList *owner, ItemListElement *elem ):
   m_owner( owner ),
   m_element( elem ),
   m_next( 0 ),
   m_prev( 0 )
{
   if ( m_owner != 0 )
   {
      m_owner->addIterator( this );
   }
}


ItemListIterator::~ItemListIterator()
{
   if ( m_owner != 0 )
      m_owner->removeIterator( this );
}


bool ItemListIterator::next()
{
   if ( m_element )
   {
      m_element = m_element->next();
      return m_element != 0;
   }

   return false;
}

bool ItemListIterator::prev()
{
   if ( m_element )
   {
      m_element = m_element->prev();
      return m_element != 0;
   }

   return false;
}


bool ItemListIterator::hasNext() const
{
   return m_element != 0 && m_element->next() != 0;
}

bool ItemListIterator::hasPrev() const
{
   return m_element != 0 && m_element->prev() != 0;
}


Item &ItemListIterator::getCurrent() const
{
   return m_element->item();
}


bool ItemListIterator::isValid() const
{
   return m_element != 0;
}


bool ItemListIterator::isOwner( void *collection ) const
{
   return m_owner == collection;
}


bool ItemListIterator::equal( const CoreIterator &other ) const
{
   if( other.isOwner( m_owner ) )
      return m_element == static_cast<const ItemListIterator *>( &other )->m_element;
   return false;
}

void ItemListIterator::invalidate()
{
   m_element = 0;
}


UserData *ItemListIterator::clone()
{
   if ( m_element == 0 )
      return 0;

   return new ItemListIterator( m_owner, m_element );
}

void ItemListIterator::setCurrentElement( ItemListElement *e )
{
   m_element = e;
}

bool ItemListIterator::erase()
{
   if ( m_owner != 0 )
   {
      return m_owner->erase( this );
   }
   return false;
}

bool ItemListIterator::insert( const Item &item )
{
   if ( m_owner != 0 )
   {
      return m_owner->insert( this, item );
   }
   return false;
}

}

/* end of itemlist.cpp */
