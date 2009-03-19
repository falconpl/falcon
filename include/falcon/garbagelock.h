/*
   FALCON - The Falcon Programming Language.
   FILE: garbagelock.h

   Garbage lock - safeguards for items in VMs.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 19 Mar 2009 19:59:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_GARBAGELOCK_H
#define FALCON_GARBAGELOCK_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS GarbageLock: public BaseAlloc
{
   GarbageLock *m_garbage_next;
   GarbageLock *m_garbage_prev;

   Item m_item;

public:

   GarbageLock( const Item &itm ):
      m_item( itm )
   {}

   const Item &item() const { return m_item; }
   Item &item() { return m_item; }

   GarbageLock *next() const { return m_garbage_next; }
   GarbageLock *prev() const { return m_garbage_prev; }
   void next( GarbageLock *next ) { m_garbage_next = next; }
   void prev( GarbageLock *prev ) { m_garbage_prev = prev; }
};

}

#endif

/* end of garbagelock.h */
