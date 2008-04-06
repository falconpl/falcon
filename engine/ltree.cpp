/*
   FALCON - The Falcon Programming Language.
   FILE: ltree.cpp

   Strong list definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Feb 26 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Strong list definitions
*/

#include <falcon/setup.h>
#include <falcon/ltree.h>

namespace Falcon {

void StrongList::push_front( SLElement *elem )
{
   elem->next( m_head );
   elem->prev(0);

   if( m_tail == 0 )
      m_tail = elem;
   else
      m_head->prev( elem );
   m_head = elem;
   elem->owner( this );
   m_size++;
}

void StrongList::push_back( SLElement *elem )
{
   elem->next( 0 );
   elem->prev( m_tail  );

   if( m_head == 0 )
      m_head = elem;
   else
      m_tail->next( elem );
   m_tail = elem;

   elem->owner( this );
   m_size++;
}

SLElement *StrongList::pop_front()
{
   SLElement *h = m_head;
   if (m_head != 0 ) {
      m_head = m_head->next();
      if ( m_head != 0 )
         m_head->prev(0);
      else
         m_tail = 0;
      m_size--;

      h->owner( 0 );
   }

   return h;
}

SLElement *StrongList::pop_back()
{
   SLElement *t = m_tail;
   if (m_tail != 0 ) {
      m_tail = m_tail->prev();
      if ( m_tail != 0 )
         m_tail->next(0);
      else
         m_head = 0;
      m_size--;

      t->owner( 0 );
   }

   return t;
}

void StrongList::remove( SLElement *elem )
{
   if ( m_size == 0 || elem->owner() != this ) return;
   if ( elem == m_head ) {
      m_head = elem->next();
   }
   else {
      elem->prev()->next( elem->next() );
   }

   if ( elem == m_tail ) {
      m_tail = elem->prev();
   }
   else {
      elem->next()->prev( elem->prev() );
   }
   m_size--;
   elem->owner( 0 );
}

}


/* end of ltree.cpp */
