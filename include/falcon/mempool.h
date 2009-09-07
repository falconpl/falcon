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

#ifndef FALCON_MEMPOOL_H
#define FALCON_MEMPOOL_H

/** \file
   Garbage basket holder.
*/

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/basealloc.h>
#include <falcon/mt.h>
#include <falcon/rampmode.h>

namespace Falcon {

class Garbageable;
class GarbageableBase;
class GarbageLock;

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

   /** Minimal generation.
      Items marked with generations lower than this are killed.
   */
   uint32 m_mingen;

   /** Alive and possibly collectable items are stored in this ring. */
   GarbageableBase *m_garbageRoot;

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
   int32 m_allocatedItems;
   uint32 m_allocatedMem;

   SysThread *m_th;
   bool m_bLive;

   Event m_eRequest;
   Mutex m_mtxa;


   /** Mutex for newly created items ring.
      - GarbageableBase::nextGarbage()
      - GarbageableBase::prevGarbage()
      - m_generation
      - m_newRoot
      - rollover()
      \note This mutex is acquired once while inside  m_mtx_vms.lock()
   */
   mutable Mutex m_mtx_newitem;

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

   /** Guard for ramp modes. */
   mutable Mutex m_mtx_ramp;
   
   mutable Mutex m_mtx_gen;

   RampMode* m_ramp[RAMP_MODE_COUNT];
   RampMode* m_curRampMode;
   int m_curRampID;

   Mutex m_mtxRequest;
   Event m_eGCPerformed;
   bool m_bRequestSweep;

   /** Mutex for locked items ring. */
   Mutex m_mtx_lockitem;

   /** Locked and non reclaimable items are stored in this ring.  */
   GarbageLock *m_lockRoot;

   /** Generation at which locked items have been marked. *
    *
    */
   uint32 m_lockGen;

   //==================================================
   // Private functions
   //==================================================

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
   
   friend class GarbageLock;
   void addGarbageLock( GarbageLock* lock );
   void removeGarbageLock( GarbageLock* lock );

public:
   enum constants {
      MAX_GENERATION = 0xFFFFFFFE,
      SWEEP_GENERATION = 0xFFFFFFFF
   };

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
   void markItem( const Item &itm );

   /** Returns the number of elements managed by this mempool. */
   int32 allocatedItems() const;

   /** Returns the current generation. */
   uint32 generation() const { return m_generation; }
   /*
   void generation( uint32 i );
   uint32 incgen();
   */
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

   /** Sets the algorithm used to dynamically configure the collection levels.
      Can be one of:
      - RAMP_MODE_STRICT_ID
      - RAMP_MODE_LOOSE_ID
      - RAMP_MODE_SMOOTH_SLOW_ID
      - RAMP_MODE_SMOOTH_FAST_ID

      Or RAMP_MODE_OFF to disable dynamic auto-adjust of collection levels.
      \param mode the mode to be set.
      \return true if the mode can be set, false if it is an invalid value.
   */
   bool rampMode( int mode );
   int rampMode() const;
   
   /** Alter the count of live items.
     For internal use.
   */
   void accountItems( int itemCount );

   void performGC();
};


}

#endif
/* end of mempool.h */
