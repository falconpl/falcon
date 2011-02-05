/*
   FALCON - The Falcon Programming Language.
   FILE: collector.h

   Garbage collector
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Feb 2009 16:08:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_COLLECTOR_H
#define FALCON_COLLECTOR_H

#include <falcon/setup.h>
#include <falcon/mt.h>
#include <falcon/rampmode.h>

namespace Falcon {

class GCToken;
class GCLock;
class Item;
class CoreClass;
class VMachine;

/** Falcon Garbage collector.
 *
 * @section coll_loop Collection mark-sweep loop thread.
 *
 * The mark-sweep loop has the follwing structure:
 * - The items currently locked are marked.
 * - The live modules are makred.
 * - The VMs registered with this collector are scanned for live items to be marked.
 * - The sweep loop reapes all the items that fall beyond the last live VM generation.
 * - The new items are marked with the CURRENT generation, and moved in the normal garbage.
 * - The generation is advanced.
 *
 * Items may be locked by active threads right after the first step, but they are
 * allocated in the new item list; so, they cannot be reaped during the current loop.
 *
*/

class FALCON_DYN_CLASS Collector: public Runnable
{

protected:
   size_t m_thresholdNormal;
   size_t m_thresholdActive;

   /** Minimal generation.
      Items marked with generations lower than this are killed.
   */
   uint32 m_mingen;

   /** Alive and possibly collectable items are stored in this ring. */
   GCToken *m_garbageRoot;

   /** Newly created and unreclaimable items are stored in this ring. */
   GCToken *m_newRoot;

   /** A place where to store tokens for recycle. */
   GCToken *m_recycleTokens;
   int32 m_recycleTokensCount;
   Mutex m_mtx_recycle_tokens;


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

   RampMode* m_ramp[RAMP_MODE_COUNT];
   RampMode* m_curRampMode;
   int m_curRampID;

   Mutex m_mtxRequest;
   Event m_eGCPerformed;
   bool m_bRequestSweep;

   /** Mutex for locked items ring. */
   Mutex m_mtx_lockitem;

   /** Locked and non reclaimable items are stored in this ring.  */
   GCLock *m_lockRoot;

   GCLock *m_recycleLock;
   int32 m_recycleLockCount;
   Mutex m_mtx_recycle_locks;

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

   void clearRing( GCToken *ringRoot );
   void rollover();
   void remark(uint32 mark);
   void electOlderVM(); // to be called with m_mtx_vms locked

   void promote( uint32 oldgen, uint32 curgen );
   void advanceGeneration( VMachine* vm, uint32 oldGeneration );
   void markLocked();
   void disposeLock( GCLock* lock );
   void disposeToken(GCToken* token);



   void addGarbageLock( GCLock* lock );
   void removeGarbageLock( GCLock* lock );

   // Gets a new or pre-allocated token
   GCToken* getToken( CoreClass* cls, void* data );

   // Marks the newly created items.
   void markNew();

public:
   enum constants {
      MAX_GENERATION = 0xFFFFFFFE,
      SWEEP_GENERATION = 0xFFFFFFFF
   };

   /** Builds a memory pool.
      Initializes all element at 0 and set buffer sizes to the FALCON default.
   */
   Collector();

   /** Destroys all the items.
      Needless to say, this must be called outside any VM.
   */
   virtual ~Collector();

   /** Called upon creation of a new VM.
      This sets the current generation of the VM so that it is unique
      among the currently living VMs.
   */
   void registerVM( VMachine *vm );

   /** Called before destruction of a VM.
      Takes also care to disengage the VM from idle VM list.
   */
   void unregisterVM( VMachine *vm );


   /** Returns the number of elements managed by this mempool. */
   int32 allocatedItems() const;

   /** Returns the current generation. */
   uint32 generation() const { return m_generation; }

   virtual void* run();

   /** Starts the parallel garbage collector. */
   void start();

   /** Stops the collector.
      The function synchronously wait for the thread to exit and sets it to 0.
   */
   void stop();
   
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


   /** Run a complete garbage collection.
    *
    * This method orders the GC to perform a complete garbage collection loop as soon as
    * possible, and then waits for the completion of that loop.
    *
    */
   void performGC();

   /**
    * Stores an entity in the garbage collector.
    *
    * The entity gets stored in the new items, and will become reclaimable
    * since the first scan loop that comes next.
    *
    * The data must be delivered to the garbage collection system with the
    * class that describes it. The collector will call CoreClass::gcmark to
    * indicate that the item holding this object is alive. When the item
    * is found dead, the collector will call CoreClass::dispose to inform
    * the class that the item is not needed by Falcon anymore.
    *
    *
    * @param cls The class that manages the data.
    * @param data An arbitrary data to be passed to the garbage collector.
    */
   void store( CoreClass* cls, void* data );


   /**
    * Stores an entity in the garbage collector and immediately locks it.
    *
    * This method stores an entity for garbage collecting, but adds an initial
    * lock so that the collector cannot reclaim it, nor any other data depending
    * from the stored entity.
    *
    * This is useful when the object is known to be needed by an external entity
    * that may be destroyed separately from Falcon activity. A locked entity
    * gets marked via CoreClass::gcmark even if not referenced in any virtual machine,
    * and gets disposed only if found unreferenced after the garbage lock is
    * removed.
    *
    * @param cls The class that manages the data.
    * @param data An arbitrary data to be passed to the garbage collector.
    */
   GCLock* storeLocked( CoreClass* cls, void* data );


   /** Locks an item.
    * @see GCLock
    */
   GCLock* lock( const Item& item );

   /** Unlocks a locked item. */
   void unlock( GCLock* lock );

};

}

#endif
/* end of collector.h */
