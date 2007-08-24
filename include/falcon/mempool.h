/*
   FALCON - The Falcon Programming Language.
   FILE: flc_mempool.h
   $Id: mempool.h,v 1.16 2007/08/12 17:44:43 jonnymind Exp $

   garbage basket class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 2 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef flc_MEMPOOL_H
#define flc_MEMPOOL_H

/** \file
   Garbage basket holder.
*/

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Garbageable;
class GarbageString;

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

   GarbageLock *next() const { return m_garbage_next; }
   GarbageLock *prev() const { return m_garbage_prev; }
   void next( GarbageLock *next ) { m_garbage_next = next; }
   void prev( GarbageLock *prev ) { m_garbage_prev = prev; }
};


/** Falcon Memory pool
   The garbage basket is the Falcon standard memory allocator. It provides newly created
   objects and memory chunks and saves them for later recycle. The garbage collector moves
   unused items and memory chunks to the basket holder. Is then responsibility of the holder
   to decide what to do about them.

   The memory pool is responsible for:
      - Allocating new memory or use recycled memory.
      - Destroy memory or save block in the recycle bins for later use.
      - Keep a track of all "live" object so that the garbage collector can detect unused memory,
        and memory leaks can be detected.
*/

class FALCON_DYN_CLASS MemPool: public BaseAlloc
{
   friend class VMachine;

protected:
   uint32 m_thresholdMemory;
   uint32 m_thresholdReclaim;
   uint32 m_setThreshold;
   uint32 m_msLimit;

   VMachine *m_owner;

   Garbageable *m_garbageRoot;
   GarbageString *m_gstrRoot;

   GarbageLock *m_availPoolRoot;

   // for gc
   byte m_status;
   uint32 m_aliveItems;
   uint32 m_aliveMem;

   uint32 m_allocatedItems;
   uint32 m_allocatedMem;
   bool m_autoClean;

   bool gcMark();
   void gcSweep();
   void markItem( const Item &itm );

   /** Prevents calling the markitem function in case of shallow items. */
   void markItemFast( const Item &itm )
   {
      if( itm.type() >= FLC_ITEM_STRING )
         markItem( itm );
   }

   byte currentMark() const { return m_status; }
   void changeMark() { m_status = m_status == 1 ? 0 : 1; }
   Garbageable *ringRoot() const { return m_garbageRoot; }

   void storeForGarbage( GarbageString *ptr );
   void storeForGarbage( Garbageable *ptr );


   /*
   To reimplement this, we need to have anti-recursion checks on item, which are
   currently being under consideration. However, I would prefer not to need to
   have this functions back, as they were meant to be used when the memory
   model wasn't complete.

   In other words, I want items to be in garbage as soon as they are created,
   and to exit when they are destroyed.

   void removeFromGarbage( GarbageString *ptr );
   void removeFromGarbage( Garbageable *ptr );

   void storeForGarbageDeep( const Item &item );
   void removeFromGarbageDeep( const Item &item );
   */

public:
   /** Builds a memory pool.
      Initializes all element at 0 and set buffer sizes to the FALCON default.
   */
   MemPool();

   /** Destroys all the items.
      Needless to say, this must be called outside any VM.
   */
   ~MemPool();

   void setOwner( VMachine *owner ) { m_owner = owner; }

   /** Destroys a garbageable element.
      \note is this useful???
   */
   void destroyGarbage( Garbageable *ptr );

   /** Destroys a garbageable element.
      \note is this useful???
   */
   void destroyGarbage( GarbageString *ptr );

   /** Return current threshold memory level.
      \see thresholdMemory( uint32 mem )
      \return current threshold memory level.
   */
   uint32 thresholdMemory() const { return m_thresholdMemory; }

   /** Set threshold memory level.
      The threshold memory level is the amount of allocated memory at which
      the GC normally considers the possibility of scan memory for
      items to be reclaimed.

      Setting it too low may cause GC to be often employed in short
      reclaim loops that will actually reduce your programs performance,
      while keeping it too high may cause too much memory to be acquired
      by the VM and/or may force GC to excessively long loops.
   */
   void thresholdMemory( uint32 mem ) { m_thresholdMemory = mem; m_setThreshold = mem; }

   /** Return current reclaim memory level.
      \see reclaimLevel( uint32 mem )
      \return current reclaim memory level.
   */
   uint32 reclaimLevel() const { return m_thresholdReclaim; }

   /** Set reclaim memory level.
      The reclaim memory level is the amount of unused memory that,
      once detected, will cause the GC to start a collection loop.

      After a mark loop, the GC may find that the allocated but unused
      memory is quite small, so small that it's actually not worth to
      perform a full GC on that.

      This level indicates how much unallocated memory the GC may
      detect before deciding to intervene ad use additional time for
      the actual collection loop.
   */
   void reclaimLevel( uint32 mem ) { m_thresholdReclaim = mem; }


   /** Perform garbage collection loop.
      Garbage collection is divided in two parts: free memory
      identification and reclaiming.
      Normally, GC would first identify memory that can be collected,
      and then decide if the memory to be collected is wide enough.
      To force memory collection even if unused memory treshold is
      not met, pass bForceReclaim true
      \param bForceReclaim true to reclaim memory no matter how
            small the memory to be reclaimed is
      \return true if some memory has been collected.
   */
   virtual bool performGC( bool bForceReclaim = false );

   /** Used by garbageable objects to update their allocation size */
   void updateAlloc( int32 sizeChange ) { m_allocatedMem += sizeChange; }
   /** Returns the size of memory managed by this mempool. */
   uint32 allocatedMem() const { return m_allocatedMem; }

   /** Returns the number of elements managed by this mempool. */
   uint32 allocatedItems() const { return m_allocatedItems; }

   /** Returns the amount of memory that the last mark loop has found alive.
      This is the memory allocated to items that are reachable from the
      items the current module structure is holding.
   */
   uint32 aliveMem() const { return m_aliveMem; }

   /** Returns the number of elements managed by this mempool. */
   uint32 aliveItems() const { return m_aliveItems; }

   /** Checks for garbage levels and eventually starts the GC loop.
      In case the garbage levels are above the threshold levels, a standard
      performGC() is called.
      The checkForGarbage() method is called periodically by the VM, and it may be
      called after functions that are known to generate much garbage.

      The autoCleanMode() method may be used to prevent this method to ever call
      performGC().
   */
   bool checkForGarbage();

   /** Return current autoclean status.
      \return true if autoclean is enabled.
   */
   bool autoCleanMode() const { return m_autoClean; }
   /** Set autoclean on threshold.
      If false, checkForGarbage() will never start a reclaim loop,
      even if allocated memory is above warning threshold.
      \param mode false to disable auto GC reclaim.
   */
   void autoCleanMode( bool mode ) { m_autoClean = mode; }

   /** Set maximum timeout for GC loop to perform.
      This value is ignored in this class, but it is available here
      so that subclasses may use it as a time limit for lengty
      GC collection loops.

      Setting timeout to zero will disable the time limit
      \param ms timeout expressed in milliseconds
   */
   void setTimeout( uint32 ms ) { m_msLimit = ms; }

   /** Return current timeout for GC loops.
      \see setTimeout
      \return timeout for GC loops expressed in milliseconds
   */
   uint32 getTimeout() const { return m_msLimit; }

   /** Locks garbage data.

      Puts the given item in the availability pool. Garbage sensible
      objects in that pool and objects reachable from them will be marked
      as available even if there isn't any VM related entity pointing to them.

      For performance reasons, a copy of the item stored in a GarbageItem
      is returned. The calling application save that pointer and pass it
      to unlock() when the item can be released.

      It is not necessary to unlock the locked items: at VM destruction
      they will be correctly destroyed.

      Both the scripts (the VM) and the application may use the data in the
      returned GarbageItem and modify it at will.

      \param locked entity to be locked.
      \return a relocable item pointer that can be used to access the deep data.
   */
   GarbageLock *lock( const Item &locked );

   /** Unlocks garbage data.
      Moves a locked garbage sensible item back to the normal pool,
      where it will be removed if it is not reachable by the VM.

      \note after calling this method, the \b locked parameter becomes
         invalid and cannot be used anymore.

      \see lock

      \param locked entity to be unlocked.
   */
   void unlock( GarbageLock *locked );
};


}

#endif
/* end of flc_mempool.h */
