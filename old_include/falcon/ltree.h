/*
   FALCON - The Falcon Programming Language.
   FILE: ltree.h

   Inlined tree linked list templates.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Feb 26 08:45:59 CET 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Inlined linked list templates.
*/

#ifndef FLC_LTREE_H
#define FLC_LTREE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

namespace Falcon {

class StrongList;

/** Element for StrongList class */

class FALCON_DYN_CLASS SLElement: public BaseAlloc
{
   SLElement *m_next;
   SLElement *m_prev;
   StrongList *m_parent;

public:

   SLElement( SLElement *prev=0, SLElement *next=0 ):
      m_next( next ),
      m_prev( prev ),
      m_parent(0)
   {}

   SLElement( StrongList *parent, SLElement *prev=0, SLElement *next=0 ):
      m_next( next ),
      m_prev( prev ),
      m_parent(parent)
   {}

   SLElement *next() const { return m_next; }
   void next( SLElement *n ) { m_next = n; }
   SLElement *prev() const { return m_prev; }
   void prev( SLElement *p ) { m_prev = p; }

   StrongList *owner() const { return m_parent; }
   void owner( StrongList *lt ) { m_parent = lt; }

   void remove();
};

/** Strong linked list class.
   This is similar to a linked list in which each element has a link to its owner.
*/
class FALCON_DYN_CLASS StrongList: public BaseAlloc
{
   SLElement *m_head;
   SLElement *m_tail;
   uint32 m_size;

public:
   StrongList():
      m_head(0),
      m_tail(0),
      m_size(0)
   {}

   void push_front( SLElement *e );

   void push_back( SLElement *e );

   SLElement *front() const { return m_head; }
   SLElement *back() const { return m_tail; }

   SLElement *pop_front();
   SLElement *pop_back();
   void remove( SLElement *e );

   uint32 size() const { return m_size; }
   bool empty() const { return m_size == 0; }
};

inline void SLElement::remove() {
   if (m_parent!=0)
      m_parent->remove( this );
}

}

#endif

/* end of llist.h */
