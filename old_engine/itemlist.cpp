/*
   FALCON - The Falcon Programming Language.
   FILE: itemlist.cpp

   List of Falcon Items
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   List of Falcon Items implementation
*/

#include <falcon/setup.h>
#include <falcon/itemlist.h>
#include <falcon/vm.h>

namespace Falcon {

ItemList::ItemList( const ItemList &l ):
    m_erasingIter(0),
    m_disposingElem(0)
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



ItemList *ItemList::clone() const
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
   ItemListElement* elem = m_tail;
   if( elem != 0 )
   {
      m_tail = m_tail->prev();

      // was the only element?
      if ( m_tail == 0 )
      {
         // delete it and update head.
         m_head = 0;
      }
      else {
         m_tail->next()->prev(0);
         m_tail->next(0);
      }
      m_size--;

      if ( m_iterList != 0)
      {
         m_disposingElem = elem;
         invalidateIteratorOnCriterion();
         m_disposingElem = 0;
      }

      delete elem;
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
   ItemListElement* elem = m_head;

   if( m_head != 0 )
   {
      m_head = m_head->next();
      
      // was the only element?
      if ( m_head == 0 )
      {
         // delete it and update head.
         m_tail = 0;
      }
      else {
         m_head->prev()->next(0);
         m_head->prev(0);
      }
      m_size--;

      if ( m_iterList != 0)
      {
         m_disposingElem = elem;
         invalidateIteratorOnCriterion();
         m_disposingElem = 0;
      }

      delete elem;
   }
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
   
   // invalidate all the iterators on this element.
   if( m_iterList )
   {
      m_disposingElem = elem;
      invalidateIteratorOnCriterion();
      m_disposingElem = 0;
   }
   
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


void ItemList::clear()
{
   invalidateAllIters();
   
   ItemListElement *h = m_head;
   while( h != 0 )
   {
      ItemListElement *nx = h->next();
      h->next(h);
      h->prev(0);
      delete h;
      h = nx;
   }
   m_head = 0;
   m_tail = 0;
   m_size = 0;
}


void ItemList::gcMark( uint32 gen )
{
   Sequence::gcMark( gen );

   // we don't have to record the mark byte, as we woudln't have been called
   // if the coreobject holding us had the right mark.

   ItemListElement *h = m_head;
   while( h != 0 )
   {
      memPool->markItem( h->item() );
      h = h->next();
   }
}


//========================================================
// Iterator implementation.
//========================================================

void ItemList::getIterator( Iterator& tgt, bool tail ) const
{  
   Sequence::getIterator( tgt, tail );
   ItemListElement* elem = tail ? m_tail : m_head;

   // may legally be 0 (for insertion at end)
   tgt.data( elem );
}


void ItemList::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   tgt.data( source.data() );
}


void ItemList::insert( Iterator &tgt, const Item &data )
{
   ItemListElement* ptr = (ItemListElement*) tgt.data();

   if ( ptr == 0 )
      append( data );
   else {
      insert( ptr, data );
      tgt.prev();
   }
}

void ItemList::erase( Iterator &tgt )
{
   ItemListElement* ptr = (ItemListElement*) tgt.data();
   if ( ptr == 0 )
   {
      throw new AccessError( ErrorParam( e_invalid_iter )
            .origin( e_orig_runtime ).extra( "ItemList::erase" ) );
   }

   m_erasingIter = &tgt;
   ItemListElement* next = erase( ptr );   
   m_erasingIter = 0;
   
   tgt.data( next );
}


bool ItemList::hasNext( const Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   return ptr != 0 && ptr->next() != 0;
}


bool ItemList::hasPrev( const Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   // ptr == 0 => at end
   return ( ptr == 0 && m_size > 0 ) || (ptr!= 0 && ptr->prev() != 0);
}


bool ItemList::hasCurrent( const Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   return ptr != 0;
}

bool ItemList::next( Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
      return false;

   ItemListElement* next = ptr->next();
   // change even if next == 0 (it's a valid iterator at end).
   iter.data( next );
   return next != 0;
}


bool ItemList::prev( Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   
   // zero means "at end", so the previous element is the tail
   if ( ptr == 0 )
   {
      ptr = m_tail;

      if ( ptr == 0 )
      {
         return false;
      }

      iter.data( ptr );
      return true;
   }

   ItemListElement* prev = ptr->prev();
   iter.data( prev );
   return prev != 0;
}

Item& ItemList::getCurrent( const Iterator &iter )
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
   {
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
            .origin( e_orig_runtime ).extra( "ItemList::getCurrent" ) );
   }

   return ptr->item();
}

Item& ItemList::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
              .origin( e_orig_runtime ).extra( "ItemList::getCurrentKey" ) );
}

bool ItemList::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.data() == second.data();
}

// Deletion criterion.
bool ItemList::onCriterion( Iterator* elem ) const
{
   return elem != m_erasingIter && elem->data() == m_disposingElem;
}


}

/* end of itemlist.cpp */
