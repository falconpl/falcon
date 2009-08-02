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

ItemList::ItemList( const ItemList &l )
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



FalconData *ItemList::clone() const
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
   if( m_tail != 0 )
   {
      m_tail = m_tail->prev();

      // was the only element?
      if ( m_tail == 0 )
      {
         // delete it and update head.
         m_head->decref();
         m_head = 0;
      }
      else {
         m_tail->next()->prev(0);
         m_tail->next()->decref();
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
         m_tail->decref();
         m_tail = 0;
      }
      else {
         m_head->prev()->next(0);
         m_head->prev()->decref();
         m_head->prev(0);
      }
      m_size--;
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

   elem->decref();
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
   ItemListElement *h = m_head;
   while( h != 0 )
   {
      ItemListElement *nx = h->next();
      h->next(0);
      h->prev(0);
      h->decref();

      h = nx;
   }
   m_head = 0;
   m_tail = 0;
   m_size = 0;

}


void ItemList::gcMark( uint32 mark )
{
   Sequence::gcMark( mark );

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
   if( ! empty() )
   {
      tgt.data( tail ? m_tail : m_head );
   }
   // else keep the 0 value
}


void ItemList::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   tgt.data( source.data() );
}

void ItemList::disposeIterator( Iterator& tgt ) const
{
   ItemListElement* ptr = (ItemListElement*) tgt.data();
   if( ptr != 0 )
      ptr->decref();
}

void ItemList::gcMarkIterator( Iterator& tgt ) const
{
   // nothing
}

void ItemList::insert( Iterator &tgt, const Item &data )
{
   ItemListElement* ptr = (ItemListElement*) tgt.data();
   if( ptr != 0 )
      ptr->decref();
}

void ItemList::erase( Iterator &tgt )
{
   ItemListElement* ptr = (ItemListElement*) tgt.data();
   if ( ptr == 0 )
   {
      tgt.invalidate();
      throw new AccessError( ErrorParam( e_invalid_iter ) );
   }

   ItemListElement* next = erase( ptr );
   if ( next != 0 )
      next->incref();

   // isolate the item
   next->next(0);
   next->prev(0);

   ptr->decref();
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
   return ptr != 0 && ptr->prev() != 0;
}


bool ItemList::next( Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
   {
      iter.invalidate();
      throw new AccessError( ErrorParam( e_invalid_iter, __LINE__ ) );
   }

   ptr->decref();
   ItemListElement* next = ptr->next();
   ptr->decref();
   if ( next != 0 )
      next->incref();

   iter.data( next );
}


bool ItemList::prev( Iterator &iter ) const
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
   {
      iter.invalidate();
      throw new AccessError( ErrorParam( e_invalid_iter, __LINE__ ) );
   }

   ItemListElement* prev = ptr->prev();
   ptr->decref();
   if ( prev != 0 )
      prev->incref();

   iter.data( prev );
}

Item& ItemList::getCurrent( const Iterator &iter )
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
   {
      throw new AccessError( ErrorParam( e_invalid_iter, __LINE__ ) );
   }

   return ptr->item();
}

Item& ItemList::getCurrentKey( const Iterator &iter )
{
   ItemListElement* ptr = (ItemListElement*) iter.data();
   if ( ptr == 0 )
   {
      throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ ) );
   }
}

bool ItemList::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.data() == second.data();
}

//========================================================

void ItemListElement::decref()
{
   if ( atomicDec( m_refCount ) == 0 )
   {
      delete this;
   }
}

}

/* end of itemlist.cpp */
