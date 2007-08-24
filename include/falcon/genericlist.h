/*
   FALCON - The Falcon Programming Language.
   FILE: genlist.h
   $Id: genericlist.h,v 1.7 2007/04/02 00:24:00 jonnymind Exp $

   Generic list - a list holding generic values.
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

#ifndef flc_genlist_H
#define flc_genlist_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/traits.h>
#include <falcon/basealloc.h>

namespace Falcon {

/** Generic list element.
*/
class FALCON_DYN_CLASS ListElement: public BaseAlloc
{
   ListElement *m_next;
   ListElement *m_previous;
   const void *m_data;

   ListElement( const void *data ):
      m_data(data)
   {
   }

   void data( const void *data ) { m_data = data; }
   void prev( ListElement *elem ) { m_previous = elem; }
   void next( ListElement *elem ) { m_next = elem; }

   friend class List;

public:
   const void *data() const { return m_data; }

   ListElement *next() const { return m_next; }
   ListElement *prev() const { return m_previous; }
};

/** Generic list.
*/
class FALCON_DYN_CLASS List: public BaseAlloc
{
   ListElement *m_head;
   ListElement *m_tail;

   void (*m_deletor)( void *);

	friend class ListTraits;
public:
   List():
      m_head(0),
      m_tail(0),
      m_deletor(0)
   {
   }

    List( void (*deletor)(void *) ):
      m_head(0),
      m_tail(0),
      m_deletor( deletor )
   {
   }

   ~List()
   {
      clear();
   }

   ListElement *begin() const { return m_head; }
   ListElement *end() const { return m_tail; }
   const void *front() const { return m_head->data(); }
   const void *back() const { return m_tail->data(); }
   bool empty() const { return m_head == 0; }

   void pushFront( const void *data );
   void pushBack( const void *data );
   void popFront();
   void popBack();
   void insertAfter( ListElement *position, const void *data );
   void insertBefore( ListElement *position, const void *data );

   ListElement *erase( ListElement *position );
   uint32 size() const;
   void clear();

   void deletor( void (*del)( void * ) ) { m_deletor = del; }
};

class ListTraits: public ElementTraits
{
public:
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

namespace traits
{
	extern FALCON_DYN_SYM ListTraits t_List;
}

}

#endif

/* end of genlist.h */
