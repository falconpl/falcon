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

class MemPool;

/* Notice that this class is implemented in mempool.cpp */

/** Protects an item from garbage collecting.

   This class safely store an item, and all its contents in case it is a
   sort of container, from being collected. The garbage locked items
   are marked as soon as they reach the lowest possible value
   in the live generation count. Extensions are granted to have their
   ::gcMark() method called when this happens, so any kind of item
   can be safely stored as the item() element.

   Once destroyed, the garbage lock releases the item, that 
   can then be immediately collected.
*/
class FALCON_DYN_CLASS GarbageLock: public BaseAlloc
{
   GarbageLock *m_garbage_next;
   GarbageLock *m_garbage_prev;

   Item m_item;

   friend class MemPool;
   GarbageLock *next() const { return m_garbage_next; }
   GarbageLock *prev() const { return m_garbage_prev; }
   void next( GarbageLock *next ) { m_garbage_next = next; }
   void prev( GarbageLock *prev ) { m_garbage_prev = prev; }

   // Constructor used to initialize the gclock ring
   GarbageLock( bool );

public:
   /** Creates an empty garbage lock.
   
       The item inside this garbage lock is nil.
   */
   GarbageLock();
   
   /** Creates a garbage lock protecting an item. */
   GarbageLock( const Item &itm );

   /** Releases the item that can now be collected. */
   ~GarbageLock();

   /** Return the guarded item (const version). */
   const Item &item() const { return m_item; }

   /** Return the guarded item.

   The returned value can be modified. For example, setting it to nil
   or to another flat value will cause the previously guarded value
   to be released, and collectible for garbage.
   */
   Item &item() { return m_item; }
};

/* Notice that this class is implemented in mempool.cpp */

}

#endif

/* end of garbagelock.h */
