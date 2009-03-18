/*
   FALCON - The Falcon Programming Language.
   FILE: mempool.h

   garbage basket class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Feb 2009 16:08:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_MEMPOOL_H
#define flc_MEMPOOL_H

/** \file
   Garbage basket holder.
*/

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/basealloc.h>
#include <falcon/mt.h>

namespace Falcon {

class Garbageable;
class GarbageableBase;

/** Storage pit for garbageable data.
   Garbage items can be removed acting directly on them.
*/
#if 0
class FALCON_DYN_CLASS PoolRing: public BaseAlloc
{
   Garbageable* m_head;

public:
   PoolRing();
   ~PoolRing();

   /**
      Adds a garbageable item to this pool ring.
   */
   void add( Garbageable * );
   void transfer( PoolRing *target );
   Garbageable *linearize();
};
#endif


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

class FALCON_DYN_CLASS MemPool: public Runnable, public BaseAlloc
{

protected:
   size_t m_thresholdNormal;
   size_t m_thresholdActive;
   
   uint32 m_setThreshold;
   uint32 m_msLimit;
   /** Minimal generation.
      Items marked with generations lower than this are killed. 
   */
   uint32 m_mingen; 

   /** Alive and possibly collectable items are stored in this ring. */
   GarbageableBase *m_garbageRoot;
   
   /** Locked and unreclaimable items are stored in this ring. */
   GarbageLock *m_lockRoot;
   
   /** Newly created and unreclaimable items are stored in this ring. */
   GarbageableBase *m_newRoot;
   
   /** Used to block prevent the pool from grabbing new garbage. */
   bool m_bNewReady;
   
   /** The machine with the oldest generation loop checked out. */
   VMachine *m_olderVM;
   
   /** Ring of VMs */
   VMachine *m_vmRing;
   
   int32 m_vmCount;
   
   /** List of VM in idle state and waiting to be inspected */
   VMachine *m_vmIdle_head;
   VMachine *m_vmIdle_tail;
   
   // for gc
   uint32 m_generation;
   int32 m_aliveItems;
   uint32 m_aliveMem;

   int32 m_allocatedItems;
   uint32 m_allocatedMem;
   bool m_autoClean;

   SysThread *m_th;
   bool m_bLive;
   
   Event m_eRequest;
   Mutex m_mtxa;
   
   
   /** Mutex for locked items ring. 
      
   */
   Mutex m_mtx_lockitem;
   
   /** Mutex for newly created items ring. 
      - GarbageableBase::nextGarbage()
      - GarbageableBase::prevGarbage()
      - m_generation
      - m_newRoot
      - rollover()
      \note This mutex is acquired once while inside  m_mtx_vms.lock()
   */
   Mutex m_mtx_newitem;
   
   /** Mutex for the VM ring structure.
      - VMachine::m_nextVM
      - VMachine::m_prevVM
      - m_vmRing
      - electOlderVM()   -- call guard
      - advanceGeneration()
   */
   Mutex m_mtx_vms;
   
   /** Mutex for the idle VM list structure. 
      Guards the linked list of VMs being in idle state.
      
      - VMachine::m_idleNext
      - VMachine::m_idlePrev
      - m_vmIdle_head
      - m_vmIdle_tail
   */
   Mutex m_mtx_idlevm;
   
   bool markVM( VMachine *vm );
   void gcSweep();
   
   /*
   To reimplement this, we need to have anti-recursion checks on item, which are
   currently being under consideration. However, I would prefer not to need to
   have this functions back, as they were meant to be used when the memory
   model wasn't complete.

   In other words, I want items to be in garbage as soon as they are created,
   and to exit when they are destroyed.

   void removeFromGarbage( String *ptr );
   void removeFromGarbage( Garbageable *ptr );

   void storeForGarbageDeep( const Item &item );
   void removeFromGarbageDeep( const Item &item );
   */
   
   void clearRing( GarbageableBase *ringRoot );
   void rollover();
   void remark(uint32 mark);
   void electOlderVM(); // to be called with m_mtx_vms locked
   
   void promote( uint32 oldgen, uint32 curgen );
   void advanceGeneration( VMachine* vm, uint32 oldGeneration );
   void markLocked();
   
   enum constants { 
      MAX_GENERATION = 0xFFFFFFFF
   };
   
public:
   /** Builds a memory pool.
      Initializes all element at 0 and set buffer sizes to the FALCON default.
   */
   MemPool();
   
   /** Destroys all the items.
      Needless to say, this must be called outside any VM.
   */
   virtual ~MemPool();

   /** Called upon creation of a new VM.
      This sets the current generation of the VM so that it is unique 
      among the currently living VMs.
   */
   void registerVM( VMachine *vm );
   
   /** Called before destruction of a VM.
      Takes also care to disengage the VM from idle VM list.
   */
   void unregisterVM( VMachine *vm );

   /** Marks an item during a GC Loop.
      This method should be called only from inside GC mark callbacks
      of class having some GC hook.
   */
   void markItem( Item &itm );

   /** Prevents calling the markitem function in case of shallow items. */
   void markItemFast( Item &itm )
   {
      if( itm.isDeep() )
         markItem( itm );
   }

 
   /** Return current threshold memory level.
      \see thresholdMemory( uint32 mem )
      \return current threshold memory level.
   */
   //uint32 thresholdMemory() const { return m_thresholdMemory; }


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

   /** Returns the current generation. */
   uint32 generation() const { return m_generation; }

   /** Stores a garbageable instance in the pool.
      Called by the Garbageable constructor to ensure accounting of this item.
   */
   void storeForGarbage( Garbageable *ptr );
   
   virtual void* run();

   /** Starts the parallel garbage collector. */
   void start();
   
   /** Stops the collector.
      The function synchronously wait for the thread to exit and sets it to 0.
   */
   void stop();
   
   /** Turns the GC safe allocation mode on.
      
      In case an "core" class object (like CoreObject, CoreDict, CoreArray, CoreString and so on)
      needs to be declared in a place where it cannot be granted that there is a working
      virtual machine, it is necessary to ask the Garbage Collector not to 
      try to collect newly allocated data.
      
      When core data is allocated inside a running VM, the GC ensures that the data
      cannot be ripped away before it reaches a safe area in a virtual machine; but
      modules or embedding applications may will to allocate garbage sensible data
      without any chance to control the idle status of the running virtual machines.
      
      To inform the GC about this fact, the safeArea(); / unsafeArea() functions are
      provided.
      
      Data should be assigned to a virtual machine or alternatively garbage locked
      before unsafeArea() is called to allow the Garbage Collector to proceed
      normally.
      
      Example:
      \code
         // ask the memory pool to delay checks on newly allocated data.
         memPool->safeArea();
         
         // Create an object instance
         CoreObject *co = myClass->createInstance();
         
         // work on the core object.
         
         // save it somewhere
         if( bCond )
            myVM->retval( co );  // assign to a non-running virtual machine
         else
            GarbageLock *safe = memPool->gcLock( co );  // save it for later usage
         
         // We're clear to proceed
         memPool->unsafeArea();
      \endcode
      
      \note Keep in mind that safe areas are global. The Garbage Collecotr won't be able to
            check for newly allocated data generated by all the running VMs in the meanwhile,
            so reduce it's use to the minimum.
   */
   void safeArea();
   
   /** Allows VM to proceed in checking newly allocated data.
      \see safeArea()
   */
   void unsafeArea();
   
   /** Declares the given VM idle.
      
      The VM may be sent to the the main memory pool garbage collector mark loop
      if it is found outdated and in need of a new marking.
      
      Set prio = true if the VM requests a priority GC. In that case, the VM
      must present itself non-idle, and the idle-ship is taken implicitly by
      the GC. The VM is notified with m_eGCPerformed being set after the complete
      loop is performed.
   */
   void idleVM( VMachine *vm, bool bPrio = false );
   
   /** Sets the normal threshold level. */
   void thresholdNormal( size_t mem ) { m_thresholdNormal = mem; }
   
   /** Sets the active threshold level. */
   void thresholdActive( size_t mem ) { m_thresholdActive = mem; }
   
   size_t thresholdNormal() const { return m_thresholdNormal; }
   
   size_t thresholdActive() const { return m_thresholdActive; }
   
   
};


}

#endif
/* end of mempool.h */
