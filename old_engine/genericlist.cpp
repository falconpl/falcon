/*
   FALCON - The Falcon Programming Language
   FILE: List.cpp

   a list holding generic values.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 15 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   a list holding generic values.
*/

#include <falcon/setup.h>
#include <falcon/setup.h>
#include <falcon/memory.h>
#include <falcon/genericlist.h>
#include <falcon/globals.h>

#include <string.h>

namespace Falcon {

void List::clear()
{
   ListElement *elem = m_head;
   if( m_deletor != 0 )
   {
      while( elem != 0 )
      {
         ListElement *toBeDeleted = elem;
         elem = elem->next();
         m_deletor( (void *) toBeDeleted->data() );
         memFree( toBeDeleted );
      }
   }
   else {
      while( elem != 0 )
      {
         ListElement *toBeDeleted = elem;
         elem = elem->next();
         memFree( toBeDeleted );
      }
   }

   m_head = m_tail = 0;
   m_size = 0;
}


void List::pushFront( const void *data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->data( data );

   if ( m_head == 0 )
   {
      m_head = m_tail = element;
      element->next(0);
   }
   else {
      element->next( m_head );
      m_head->prev( element );
      m_head = element;
   }

   m_size++;
   element->prev( 0 );
}

void List::pushBack( const void *data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->data( data );

   if ( m_head == 0 )
   {
      m_head = m_tail = element;
      element->prev(0);
   }
   else {
      element->prev( m_tail );
      m_tail->next( element );
      m_tail = element;
   }

   m_size++;
   element->next( 0 );
}


void List::pushFront( uint32 data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->iData( data );

   if ( m_head == 0 )
   {
      m_head = m_tail = element;
      element->next(0);
   }
   else {
      element->next( m_head );
      m_head->prev( element );
      m_head = element;
   }

   m_size++;
   element->prev( 0 );
}

void List::pushBack( uint32 data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->iData( data );

   if ( m_head == 0 )
   {
      m_head = m_tail = element;
      element->prev(0);
   }
   else {
      element->prev( m_tail );
      m_tail->next( element );
      m_tail = element;
   }

   m_size++;
   element->next( 0 );
}

void List::popFront()
{
   if ( m_head == 0 )
      return;

   ListElement *element = m_head;
   m_head = m_head->next();
   if( m_head != 0 )
      m_head->prev( 0 );
   else
      m_tail = 0;

   if( m_deletor != 0 ) {
      m_deletor( (void *) element->data() );
   }

   m_size--;
   memFree( element );
}

void List::popBack()
{
   if ( m_tail == 0 )
      return;

   ListElement *element = m_tail;
   m_tail = m_tail->prev();
   if( m_tail == 0 )
      m_head = 0;
   else
      m_tail->next(0);

   if( m_deletor != 0 ) {
      m_deletor( (void *) element->data() );
   }

   m_size--;
   memFree( element );
}

void List::insertAfter( ListElement *position, const void *data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->data( data );

   element->next( position->next() );
   element->prev( position );
   if( position->next() != 0 )
   {
      position->next()->prev( element );
   }

   position->next( element );

   m_size++;
   if ( position == m_tail )
      m_tail = element;
}

void List::insertBefore( ListElement *position, const void *data )
{
   ListElement *element = (ListElement *) memAlloc( sizeof( ListElement ) );
   element->data( data );

   element->next( position );
   element->prev( position->prev() );
   if( position->prev() != 0 )
   {
      position->prev()->next( element );
   }

   position->prev( element );

   m_size++;
   if ( position == m_head )
      m_head = element;
}

ListElement *List::erase( ListElement *position )
{
   ListElement *ret = position->next();
   if( position == m_head )
   {
      // could be 0 if m_head == m_tail
      m_head = ret;
      if ( ret != 0 )
      {
         ret->prev(0);
      }
      else
         m_tail = 0;
   }
   else if ( position == m_tail )
   {
      // here, ret is 0 for sure.
      m_tail = position->prev();
      m_tail->next(0);
   }
   else {
      // normal case
      position->prev()->next( ret );
      ret->prev( position->prev() );
   }

   if( m_deletor != 0 )
      m_deletor( (void *) position->data() );

   m_size--;
   memFree( position );
   return ret;
}



uint32 ListTraits::memSize() const
{
	return sizeof( List );
}

void  ListTraits::init( void *itemZone ) const
{
	List *list = (List *) itemZone;
	list->m_head = list->m_tail = 0;
	list->m_deletor = 0;
}

void ListTraits::copy( void *targetZone, const void *sourceZone ) const
{
	memcpy( targetZone, sourceZone, sizeof( List ) );
}

int ListTraits::compare( const void *first, const void *second ) const
{
	return -1;
}

void ListTraits::destroy( void *item ) const
{
	List *list = (List *) item;
	list->clear();
}

bool ListTraits::owning() const
{
	return true;
}


namespace traits
{
	FALCON_DYN_SYM ListTraits &t_List() { static ListTraits lt; return lt; }
}

}


/* end of List.cpp */
