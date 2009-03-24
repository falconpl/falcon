/*
   FALCON - The Falcon Programming Language.
   FILE: mempool.cpp

   Memory management system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-03

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/coreobject.h>
#include <falcon/carray.h>
#include <falcon/corefunc.h>
#include <falcon/corerange.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/membuf.h>
#include <falcon/garbagepointer.h>

#define GC_IDLE_TIME 250

#if 0
#define TRACE printf
#include <stdio.h>
#else
   #define TRACE(...)
#endif

// By default, 1MB
#define TEMP_MEM_THRESHOLD 1000000
namespace Falcon {

MemPool* memPool = 0;

MemPool::MemPool():
   m_mingen( 0 ),
   m_bNewReady( true ),
   m_olderVM( 0 ),
   m_vmRing(0),
   m_vmCount(0),
   m_vmIdle_head( 0 ),
   m_vmIdle_tail( 0 ),
   m_generation( 0 ),
   m_allocatedItems( 0 ),
   m_allocatedMem( 0 ),
   m_th(0),
   m_bLive(false)
{
   m_vmRing = 0;

   // use a ring for garbage items.
   m_garbageRoot = new GarbageableBase;
   m_garbageRoot->nextGarbage( m_garbageRoot );
   m_garbageRoot->prevGarbage( m_garbageRoot );

   // separate the newly allocated items to allow allocations during sweeps.
   m_newRoot = new GarbageableBase;
   m_newRoot->nextGarbage( m_newRoot );
   m_newRoot->prevGarbage( m_newRoot );

   m_thresholdNormal = TEMP_MEM_THRESHOLD;
   m_thresholdActive = TEMP_MEM_THRESHOLD*3;

   // fill the ramp algorithms
   m_ramp[RAMP_MODE_STRICT_ID] = new RampStrict;
   m_ramp[RAMP_MODE_LOOSE_ID] = new RampLoose;
   m_ramp[RAMP_MODE_SMOOTH_SLOW_ID] = new RampSmooth( 2.6 );
   m_ramp[RAMP_MODE_SMOOTH_FAST_ID] = new RampSmooth( 6.5 );

   rampMode( DEFAULT_RAMP_MODE );
}


MemPool::~MemPool()
{
   // ensure the thread is down.
   stop();

   clearRing( m_newRoot );
   clearRing( m_garbageRoot );

   // VMs are not mine, and they should be already dead since long.
   for( uint32 ri = 0; ri < RAMP_MODE_COUNT; ri++ )
      delete m_ramp[ri];
}


bool MemPool::rampMode( int mode )
{
   if( mode == RAMP_MODE_OFF )
   {
      m_mtx_ramp.lock();
      m_curRampID = mode;
      m_curRampMode = 0;
      m_mtx_ramp.unlock();
      return true;
   }
   else
   {
      if( mode >= 0 && mode < RAMP_MODE_COUNT )
      {
         m_mtx_ramp.lock();
         m_curRampID = mode;
         m_curRampMode = m_ramp[mode];
         m_curRampMode->reset();
         m_mtx_ramp.unlock();
         return true;
      }
   }

   return false;
}


int MemPool::rampMode() const
{
   m_mtx_ramp.lock();
   int mode = m_curRampID;
   m_mtx_ramp.unlock();
   return mode;
}


void MemPool::safeArea()
{
   m_mtx_newitem.lock();
   m_bNewReady = false;
   m_mtx_newitem.unlock();
}


void MemPool::unsafeArea()
{
   m_mtx_newitem.lock();
   m_bNewReady = true;
   m_mtx_newitem.unlock();
}


void MemPool::registerVM( VMachine *vm )
{
   vm->m_idlePrev = vm->m_idleNext = 0;

   m_mtx_vms.lock();
   vm->m_generation = ++m_generation; // rollover detection in run()

   int data = 0;
   ++m_vmCount;
   if ( m_vmCount > 2 )
   {
      data = 2;
   }
   else if ( m_vmCount > 1 )
   {
      data = 1;
   }

   if ( m_vmRing == 0 )
   {
      m_vmRing = vm;
      vm->m_nextVM = vm;
      vm->m_prevVM = vm;

      m_mingen = vm->m_generation;
      m_olderVM = vm;
   }
   else {
      vm->m_prevVM = m_vmRing;
      vm->m_nextVM = m_vmRing->m_nextVM;
      m_vmRing->m_nextVM->m_prevVM = vm;
      m_vmRing->m_nextVM = vm;

      // also account for older VM.
      if ( m_mingen == vm->m_generation )
      {
         m_olderVM = vm;
      }
   }

   m_mtx_vms.unlock();
}


void MemPool::unregisterVM( VMachine *vm )
{
   m_mtx_vms.lock();

   // disengage
   vm->m_nextVM->m_prevVM = vm->m_prevVM;
   vm->m_prevVM->m_nextVM = vm->m_nextVM;

   // was this the ring top?
   if ( m_vmRing == vm )
   {
      m_vmRing = m_vmRing->m_nextVM;
      // still the ring top? -- then the ring is empty
      if( m_vmRing == vm )
         m_vmRing = 0;
   }

   --m_vmCount;

   // is this the oldest VM? -- then we got to elect a new one.
   if( vm == m_olderVM )
   {
      electOlderVM();
   }

   m_mtx_vms.unlock();

   // eventually disengage from the idle list -- useful because we may unsuscribe due to destruction.
   m_mtx_idlevm.lock();
   if( m_vmIdle_head == vm )
   {
      m_vmIdle_head = vm->m_idleNext;
   }

   if( m_vmIdle_tail == vm )
   {
      m_vmIdle_tail = vm->m_idlePrev;
   }

   if ( vm->m_idleNext != 0 )
   {
      vm->m_idleNext->m_idlePrev = vm->m_idlePrev;
   }

   if ( vm->m_idlePrev != 0 )
   {
      vm->m_idlePrev->m_idleNext = vm->m_idleNext;
   }

   vm->m_idlePrev = vm->m_idleNext = 0;
   m_mtx_idlevm.unlock();
   // great; now the main thread will perform a sweep loop on its own the next time it is alive.
}


// WARNING -- this must be called with m_mtx_vms locked
void MemPool::electOlderVM()
{
   // Nay, we don't have any VM.
   if ( m_vmRing == 0 )
   {
      m_olderVM = 0;
   }
   else
   {
      VMachine *vmc = m_vmRing;
      m_mingen = vmc->m_generation;
      m_olderVM = vmc;
      vmc = vmc->m_nextVM;

      while( vmc != m_vmRing )
      {
         if ( vmc->m_generation < m_mingen )
         {
            m_mingen = vmc->m_generation;
            m_olderVM = vmc;
         }
         vmc = vmc->m_nextVM;
      }
   }
}


void MemPool::clearRing( GarbageableBase *ringRoot )
{
   // delete the garbage ring.
   GarbageableBase *ge = ringRoot->nextGarbage();
   while( ge != ringRoot )
   {
      GarbageableBase *gnext = ge->nextGarbage();
      if ( ! ge->finalize() )
         delete ge;
      ge = gnext;
   }

   delete ge;
}


void MemPool::storeForGarbage( Garbageable *ptr )
{
   // We mark newly created items as the maximum possible value
   // so they can't be reclaimed until marked at least once.
   ptr->mark( MAX_GENERATION );

   m_mtx_newitem.lock();
   m_allocatedItems++;

   ptr->prevGarbage( m_newRoot );
   ptr->nextGarbage( m_newRoot->nextGarbage() );
   m_newRoot->nextGarbage()->prevGarbage( ptr );
   m_newRoot->nextGarbage( ptr );
   m_mtx_newitem.unlock();
}


bool MemPool::markVM( VMachine *vm )
{
   vm->markLocked();

   // presume that all the registers need fresh marking
   markItemFast( vm->regA() );
   markItemFast( vm->regB() );
   markItemFast( vm->self() );

   // Latch and latcher are not necessary here because they must exist elsewhere.
   markItemFast( vm->regBind() );
   markItemFast( vm->regBindP() );

   // mark all the messaging system.
   vm->markSlots( generation() );

   // mark the global symbols
   // When generational gc will be on, this won't be always needed.
   MapIterator iter = vm->liveModules().begin();
   while( iter.hasCurrent() )
   {
      LiveModule *currentMod = *(LiveModule **) iter.currentValue();
      // We must mark the current module.
      currentMod->mark( generation() );

      ItemVector *current = &currentMod->globals();
      for( uint32 j = 0; j < current->size(); j++ )
         markItemFast( current->itemAt( j ) );

      current = &currentMod->wkitems();
      for( uint32 k = 0; k < current->size(); k++ )
         markItemFast( current->itemAt( k ) );

      iter.next();
   }

   // mark all the items in the coroutines.
   ListElement *ctx_iter = vm->getCtxList()->begin();
   uint32 pos;
   ItemVector *stack;
   while( ctx_iter != 0 )
   {
      VMContext *ctx = (VMContext *) ctx_iter->data();

      markItemFast( ctx->regA() );
      markItemFast( ctx->regB() );
      markItemFast( ctx->latcher() );

      stack = ctx->getStack();
      for( pos = 0; pos < stack->size(); pos++ ) {
         // an invalid item marks the beginning of the call frame
         if ( stack->itemAt( pos ).type() == FLC_ITEM_INVALID )
            pos += VM_FRAME_SPACE - 1; // pos++
         else
            markItemFast( stack->itemAt( pos ) );
      }

      ctx_iter = ctx_iter->next();
   }

   return true;
}

void MemPool::markItem( Item &item )
{
   switch( item.type() )
   {
      case FLC_ITEM_REFERENCE:
      {
         GarbageItem *gi = item.asReference();
         if( gi->mark() != generation() ) {
            gi->mark( generation() );
            markItemFast( gi->origin() );
         }
      }
      break;

      case FLC_ITEM_FUNC:
         if ( item.asFunction()->isValid() )
         {
            if( item.asFunction()->mark() != generation() )
            {
               item.asFunction()->mark( generation() );
            }
         }
         else
            item.setNil();
         break;

      case FLC_ITEM_RANGE:
         item.asRange()->mark( generation() );
         break;

      case FLC_ITEM_LBIND:
         if ( item.asFBind() != 0 )
         {
            GarbageItem *gi = item.asFBind();
            if ( gi->mark() != generation() )
               gi->mark( generation() );
         }
         // fallback to string for the name part

      case FLC_ITEM_STRING:
         {
            if( item.asString()->isCore() )
            {
               StringGarbage *gs = &item.asCoreString()->garbage();
               if ( gs->mark() != generation() )
               {
                  gs->mark( generation() );
               }
            }
         }
      break;

      case FLC_ITEM_GCPTR:
         item.asGCPointerShell()->mark( generation() );
         break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *array = item.asArray();
         if( array->mark() != generation() ) {
            array->mark(generation());
            for( uint32 pos = 0; pos < array->length(); pos++ ) {
               markItemFast( array->at( pos ) );
            }

            // mark also the bindings
            if ( array->bindings() != 0 )
            {
               CoreDict *cd = array->bindings();
               if( cd->mark() != generation() ) {
                  cd->mark( generation() );
                  Item key, value;
                  cd->traverseBegin();
                  while( cd->traverseNext( key, value ) )
                  {
                     markItemFast( key );
                     markItemFast( value );
                  }
               }
            }

            // and also the table
            if ( array->table() != 0 )
            {
               array->table()->mark( generation() );
            }
         }
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *co = item.asObjectSafe();
         if( co->mark() != generation() )
         {
            co->mark( generation() );
            co->gcMark( generation() );
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *cd = item.asDict();
         if( cd->mark() != generation() ) {
            cd->mark( generation() );
            Item key, value;
            cd->traverseBegin();
            while( cd->traverseNext( key, value ) )
            {
               markItemFast( key );
               markItemFast( value );
            }
         }
      }
      break;

      case FLC_ITEM_METHOD:
      {
         // if the item isn't alive, give it the death blow.
         if ( ! item.asMethodFunc()->isValid() )
            item.setNil();
         else
         {
            if( item.asMethodFunc()->mark() != generation() )
            {
               item.asMethodFunc()->mark( generation() );
            }
            Item self;
            item.getMethodItem( self );
            markItem( self );
         }
      }
      break;

      case FLC_ITEM_CLSMETHOD:
      {
         CoreObject *co = item.asMethodClassOwner();
         if( co->mark() != generation() ) {
            co->mark( generation() );
            // mark all the property values.
            co->gcMark( generation() );
         }

         CoreClass *cls = item.asMethodClass();
         if( cls->mark() != generation() ) {
            cls->mark( generation() );
            markItemFast( cls->constructor() );
            for( uint32 i = 0; i <cls->properties().added(); i++ ) {
               markItemFast( *cls->properties().getValue(i) );
            }
         }
      }
      break;

      case FLC_ITEM_CLASS:
      {
         CoreClass *cls = item.asClass();
         if( cls->mark() != generation() ) {
            cls->mark( generation() );
            markItemFast( cls->constructor() );
            for( uint32 i = 0; i <cls->properties().added(); i++ ) {
               markItemFast( *cls->properties().getValue(i) );
            }
         }
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = item.asMemBuf();
         if ( mb->mark() != generation() )
         {
            mb->mark( generation() );
            CoreObject *co = item.asMemBuf()->dependant();
            // small optimization; resolve the problem here instead of looping again.
            if( co != 0 && co->mark() != generation() )
            {
               co->mark( generation() );
               co->gcMark( generation() );
            }
         }
      }
      break;

      // all the others are shallow items; already marked
   }

}


void MemPool::gcSweep()
{
   TRACE( "Sweeping %d\n", gcMemAllocated() );
   m_mtx_ramp.lock();
   RampMode *rm = m_curRampMode;
   if( m_curRampMode != 0 )
   {
      rm->onScanInit();
   }
   m_mtx_ramp.unlock();

   int32 killed = 0;
   GarbageableBase *ring = m_garbageRoot->nextGarbage();
   while( ring != m_garbageRoot )
   {
      if ( ring->mark() < m_mingen )
      {
         killed++;
         ring->nextGarbage()->prevGarbage( ring->prevGarbage() );
         ring->prevGarbage()->nextGarbage( ring->nextGarbage() );
         GarbageableBase *dropped = ring;
         ring = ring->nextGarbage();
         if( ! dropped->finalize() )
            delete dropped;
      }
      else {
         ring = ring->nextGarbage();
      }
   }

   TRACE( "Sweeping complete %d\n", gcMemAllocated() );
   m_mtx_newitem.lock();
   fassert( killed <= m_allocatedItems );
   m_allocatedItems -= killed;
   m_mtx_newitem.unlock();

   m_mtx_ramp.lock();
   rm->onScanComplete();
   m_thresholdActive = m_curRampMode->activeLevel();
   m_thresholdNormal = m_curRampMode->normalLevel();
   m_mtx_ramp.unlock();
}

int32 MemPool::allocatedItems() const
{
   m_mtx_newitem.lock();
   int32 size = m_allocatedItems;
   m_mtx_newitem.unlock();

   return size;
}


//===================================================================================
// MT functions
//

void MemPool::idleVM( VMachine *vm, bool bPrio )
{
   m_mtx_idlevm.lock();
   if ( bPrio )
   {
      vm->m_bPirorityGC = true;
   }

   if ( vm->m_idleNext != 0 || vm->m_idlePrev != 0 || vm == m_vmIdle_head)
   {
      // already waiting
      m_mtx_idlevm.unlock();
      return;
   }

   vm->m_idleNext = 0;
   if ( m_vmIdle_head == 0 )
   {
      m_vmIdle_head = vm;
      m_vmIdle_tail = vm;
      vm->m_idlePrev = 0;
   }
   else {
      m_vmIdle_tail->m_idleNext = vm;
      vm->m_idlePrev = m_vmIdle_tail;
      m_vmIdle_tail = vm;
   }

   m_mtx_idlevm.unlock();
   // wake up if we're waiting.
   m_eRequest.set();
}

void MemPool::start()
{
   if ( m_th == 0 )
   {
      m_bLive = true;
      m_th = new SysThread( this );
      m_th->start();
   }
}

void MemPool::stop()
{
   if ( m_th != 0 )
   {
      m_bLive = false;
      m_eRequest.set();
      void *dummy;
      m_th->join( dummy );
      m_th = 0;
   }
}

void* MemPool::run()
{
   uint32 oldGeneration = m_generation;
   uint32 oldMingen = m_mingen;
   bool bMoreWork;

   while( m_bLive )
   {
      bMoreWork = false;

      // first, detect the operating status.
      size_t memory = gcMemAllocated();
      int state = memory >= m_thresholdActive ? 2 :      // active mode
                  memory >= m_thresholdNormal ? 1 :      // normal mode
                  0;                                     // dormient mode

      TRACE( "Working %d (in mode %d) \n", gcMemAllocated(), state );

      // put the new ring in the garbage ring
      m_mtx_newitem.lock();
      // Are we in a safe area?
      if ( m_bNewReady )
      {
         GarbageableBase* newRingFront = m_newRoot->nextGarbage();
         if( newRingFront != m_newRoot )
         {
            GarbageableBase* newRingBack = m_newRoot->prevGarbage();

            // disengage the chain from the new garbage thing
            m_newRoot->nextGarbage( m_newRoot );
            m_newRoot->prevGarbage( m_newRoot );
            // we can release the chain
            m_mtx_newitem.unlock();

            // and now, store the disengaged ring in the standard reclaimable garbage ring.
            TRACE( "Storing the garbage new ring in the normal ring\n" );
            newRingFront->prevGarbage( m_garbageRoot );
            newRingBack->nextGarbage( m_garbageRoot->nextGarbage() );
            m_garbageRoot->nextGarbage()->prevGarbage( newRingBack );
            m_garbageRoot->nextGarbage( newRingFront );
         }
         else
            m_mtx_newitem.unlock();
      }
      else {
         m_mtx_newitem.unlock();
         TRACE( "Skipping new ring inclusion due to safe area lock.\n" );
      }

      // if we're in active mode, send a block request to all the enabled vms.
      if ( state == 2 )
      {
         m_mtx_vms.lock();
         VMachine *vm = m_vmRing;
         if ( vm != 0 )
         {
            if ( vm->isGcEnabled() )
            {
               TRACE( "Activating blocking request vm %p\n", vm );
               vm->baton().block();
            }
            vm = vm->m_nextVM;
            while( vm != m_vmRing )
            {
               if ( vm->isGcEnabled() )
               {
                  TRACE( "Activating blocking request vm %p\n", vm );
                  vm->baton().block();
               }
               vm = vm->m_nextVM;
            }
         }
         m_mtx_vms.unlock();
      }

      VMachine* vm = 0;
      bool bPriority = false;

      // In all 3 the modes, we must clear the idle queue, so let's do that.
      m_mtx_idlevm.lock();
      if( m_vmIdle_head != 0 )
      {
         // get the first VM to be processed.
         vm = m_vmIdle_head;
         m_vmIdle_head = m_vmIdle_head->m_idleNext;
         if ( m_vmIdle_head == 0 )
            m_vmIdle_tail = 0;
         else
            bMoreWork = true;
         vm->m_idleNext = 0;
         vm->m_idlePrev = 0;

         // dormient or not, we must work this VM on priority scans.
         bPriority = vm->m_bPirorityGC;
         vm->m_bPirorityGC = false;

         // if we're dormient, just empty the queue.
         if ( state == 0 && ! bPriority )
         {
            // this to discard block requets.
            vm->baton().unblock();
            m_mtx_idlevm.unlock();

            TRACE( "Discarding idle vm %p\n", vm );
            continue;
         }

         // mark the idle VM if we're not dormient.
         // ok, we need to reclaim some memory.
         // (try to acquire only if this is not a priority scan).
         if ( ! bPriority && ! vm->baton().tryAcquire() )
         {
            m_mtx_idlevm.unlock();
            TRACE( "Was going to mark vm %p, but forfaited\n", vm );
            // oh damn, we lost the occasion. The VM is back alive.
            continue;
         }

         m_mtx_idlevm.unlock();

         m_mtx_vms.lock();
         // great; start mark loop -- first, set the new generation.
         advanceGeneration( vm, oldGeneration );
         m_mtx_vms.unlock();

         TRACE( "Marking idle vm %p \n", vm );

         // and then mark
         markVM( vm );
         // should notify now?
         if ( bPriority )
         {
            if ( ! vm->m_bWaitForCollect )
            {
               bPriority = false; // disable the rest.
               vm->m_eGCPerformed.set();
            }
         }
         else
         {
            // the VM is now free to go -- but it is not declared idle again.
            vm->baton().releaseNotIdle();
         }
      }
      else
      {
         m_mtx_idlevm.unlock();
      }

      m_mtx_vms.lock();

      // Mark of idle VM complete. See if it's useful to promote the last vm.
      if ( state > 0 && ( m_generation - m_mingen > (unsigned) m_vmCount ) )
      {
         if ( m_olderVM != 0 )
         {
            if( m_olderVM->baton().tryAcquire() )
            {
               VMachine *vm = m_olderVM;
               // great; start mark loop -- first, set the new generation.
               advanceGeneration( vm, oldGeneration );
               m_mtx_vms.unlock();

               TRACE( "Marking oldest vm %p \n", vm );
               // and then mark
               markVM( vm );
               // the VM is now free to go.
               vm->baton().releaseNotIdle();
            }
            else
            {
               m_mtx_vms.unlock();
            }
         }
         else
         {
            m_mtx_vms.unlock();
         }
      }
      else
         m_mtx_vms.unlock();

      // if we have to sweep (we can claim something only if the lower VM has moved).
      if ( oldMingen != m_mingen || bPriority )
      {
         gcSweep();
         // should we notify about the sweep being complete?
         if ( bPriority )
         {
            fassert( vm != 0 );
            vm->m_eGCPerformed.set();
         }
      }

      oldGeneration = m_generation;  // spurious read is ok here (?)
      oldMingen = m_mingen;

      // if we have nothing to do, we shall wait a bit.
      if( ! bMoreWork )
      {
         TRACE( "Waiting GC idle time\n" );
         m_eRequest.wait(GC_IDLE_TIME);
      }
   }

   TRACE( "Stopping %d \n", gcMemAllocated() );
   return 0;
}


// to be called with m_mtx_vms locked
void MemPool::advanceGeneration( VMachine* vm, uint32 oldGeneration )
{
   uint32 curgen = ++m_generation;

   // detect here rollover.
   if ( curgen < oldGeneration || curgen == MAX_GENERATION )
   {
      curgen = m_generation = m_vmCount+1;
      // perform rollover
      rollover();

      // re-mark everything
      remark( curgen );

      // as we have remarked everything, there's nothing we can do
      // but wait for the next occasion to do some collection.
      return;
   }

   vm->m_generation = curgen;

   // Now that we have marked it, if this was the oldest VM, we need to elect the new oldest vm.
   if ( vm == m_olderVM )
   {
      // calling it with mtx_vms locked
      electOlderVM();
   }
}


// WARNING: Rollover is to be called with m_mtx_vms locked.
void MemPool::rollover()
{
   // Sets the minimal VM.
   m_mingen = 1;
   m_olderVM = m_vmRing;
   m_olderVM->m_generation = 1;

   // ramp up the other VMS
   uint32 curgen = 1;
   VMachine* vm = m_vmRing->m_nextVM;
   while( vm != m_vmRing )
   {
      vm->m_generation = ++curgen;
      vm = vm->m_nextVM;
   }
}


void MemPool::remark( uint32 curgen )
{
   GarbageableBase* gc = m_garbageRoot->nextGarbage();
   while( gc != m_garbageRoot )
   {
      // Don't mark objects that are still unassigned.
      if( gc->mark() != MAX_GENERATION )
         gc->mark( curgen );

      gc = gc->nextGarbage();
   }
}

void MemPool::promote( uint32 oldgen, uint32 curgen )
{
   GarbageableBase* gc = m_garbageRoot->nextGarbage();
   while( gc != m_garbageRoot )
   {
      if( gc->mark() == oldgen )
         gc->mark( curgen );
      gc = gc->nextGarbage();
   }
}


}

/* end of mempool.cpp */
