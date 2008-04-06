/*
   FALCON - The Falcon Programming Language.
   FILE: flc_itempage.h

   Definition of the page that holds items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Definition of a segregated allocation page.
   A SAP is a memory page where all the allocated objects has the same size (and possibly are of the
   same type).

   The SAP is a template class that is instantiated by providint the size of the items at compile
   time.
*/

#ifndef flc_flc_spage_H
#define flc_flc_spage_H

#include <falcon/heap.h>
#include <falcon/types.h>
#include <falcon/llist.h>

namespace Falcon {

/**
   \note This is old code to be removed.
*/
class MemPage:public LinkableElement
{
protected:
   uint16 m_lastFree;

public:

   MemPage( MemPage *prev=0, MemPage *next=0 ):
      LinkableElement( prev, next )
   {}

   void *operator new( size_t ) throw() { return HeapMem::getPage(); }
   void *operator new( size_t, void *pos ) throw() { return pos; }
   void operator delete( void *page ) { HeapMem::freePage( page ); }
};

class MemPageList: public LinkedList< MemPage > {};

/**  Definition of a segregated allocation page.
   A SAP is a memory page where all the allocated objects has the same size (and possibly are of the
   same type).

   The SAP is a template class that is instantiated by providint the size of the items at compile
   time.
   \note This is old code to be removed.

*/
template <class _T>
class SegregatedPage: public MemPage
{
public:

   SegregatedPage( SegregatedPage<_T> *prev=0, SegregatedPage<_T> *next=0 ):
      MemPage( prev, next )
   {
      m_lastFree = sizeof( SegregatedPage<_T> );
   }

   void *nextItem() {
      if ( m_lastFree + sizeof(_T) > PAGE_SIZE )
         return 0;
      void *data= (void *)( ((char *) this) + m_lastFree );
      m_lastFree += sizeof(_T);
      return data;
   }

   void backItem() {
      if ( m_lastFree > sizeof(SegregatedPage<_T>) )
         m_lastFree -= sizeof( _T );
   }

   void resetPage() { m_lastFree = sizeof( SegregatedPage<_T> ); }

   _T *currentItem()
   {
      if ( m_lastFree == sizeof( SegregatedPage<_T> ) )
         return 0;
      return ((_T *)  (((char *) this)+m_lastFree)) -1;
   }

   _T *firstItem() {
      return (_T *) ((char*)this + sizeof( SegregatedPage<_T> ) );
   }

   _T *lastItem() {
      return ((_T *) ((char*)this + sizeof( SegregatedPage<_T> ))) +
         ( (PAGE_SIZE - sizeof( SegregatedPage<_T> )) / sizeof( _T) );
   }
};

}

#endif

/* end of flc_spage.h */
