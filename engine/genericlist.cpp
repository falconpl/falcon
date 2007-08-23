/*
   FALCON - The Falcon Programming Language
   FILE: List.cpp
   $Id: genericlist.cpp,v 1.6 2007/03/08 14:31:44 jonnymind Exp $

   a list holding generic values.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 15 2006
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
   a list holding generic values.
*/

#include <falcon/setup.h>
#include <falcon/setup.h>
#include <falcon/memory.h>
#include <falcon/genericlist.h>

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
}

uint32 List::size() const
{
   uint32 ret = 0;
   ListElement *elem = m_head;
   while( elem != 0 )
   {
      ret++;
      elem = elem->next();
   }

   return ret;
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
	FALCON_DYN_SYM ListTraits t_List;
}

}


/* end of List.cpp */
