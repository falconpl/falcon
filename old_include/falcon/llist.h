/*
   FALCON - The Falcon Programming Language.
   FILE: llist.h

   Inlined linked list templates.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven dic 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Inlined linked list templates.
*/

#ifndef FLC_LLIST_H
#define FLC_LLIST_H
#include <falcon/types.h>

namespace Falcon {

/** Element for LinkedList class */

class LinkableElement
{
   LinkableElement *m_next;
   LinkableElement *m_prev;

public:

   LinkableElement( LinkableElement *prev=0, LinkableElement *next=0 ):
      m_next( next ),
      m_prev( prev )
   {}

   LinkableElement *next() const { return m_next; }
   void next( LinkableElement *n ) { m_next = n; }
   LinkableElement *prev() const { return m_prev; }
   void prev( LinkableElement *p ) { m_prev = p; }
};

/** Linked list class.
   This class is quite similar to a STL linked list; the main difference is that
   the elements that are in the list have the linked list pointers inside them;
   this is useful when is not advisable to have external pointers holding the
   items, and the control of allocated memory must be inside the linked item
   themselves.

   Class _T must be a subclass of Falcon::LinkableElement or the compilation
   of this module will fail.
*/
template< class _T >
class LinkedList
{
   LinkableElement *m_head;
   LinkableElement *m_tail;
   uint32 m_size;

public:
   LinkedList():
      m_head(0),
      m_tail(0),
      m_size(0)
   {}

   void push_front( _T *e )
   {
      LinkableElement *elem = static_cast< LinkableElement *>( e );
      elem->next( m_head );
      elem->prev(0);
      m_head = elem;
      if( m_tail == 0 )
         m_tail = elem;
      m_size++;
   }

   void push_back( _T *e )
   {
      LinkableElement *elem = static_cast< LinkableElement *>( e );
      elem->next( 0 );
      elem->prev( m_tail  );
      m_tail = elem;
      if( m_head == 0 )
         m_head = elem;
      m_size++;
   }

   _T *front() { return static_cast<_T *>( m_head ); }
   _T *back() { return static_cast<_T *>( m_tail ); }

   _T *pop_front()
   {
      _T *h = static_cast<_T *>( m_head );
      if (m_head != 0 ) {
         m_head = m_head->next();
         if ( m_head != 0 )
            m_head->prev(0);
         else
            m_tail = 0;
         m_size--;
      }
      return h;
   }

   _T *pop_back()
   {
      _T *t = static_cast<_T *>( m_tail );
      if (m_tail != 0 ) {
         m_tail = m_tail->prev();
         if ( m_tail != 0 )
            m_tail->next(0);
         else
            m_head = 0;
         m_size--;
      }
      return t;
   }

   void remove( _T * e ) {
      LinkableElement *elem = static_cast< LinkableElement *>( e );
      if ( m_size == 0 ) return;
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
   }

   uint32 size() const { return m_size; }
   bool empty() const { return m_size == 0; }
};

}

#endif

/* end of llist.h */
