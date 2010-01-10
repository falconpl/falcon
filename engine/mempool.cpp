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

#include <falcon/trace.h>

#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/coreobject.h>
#include <falcon/carray.h>
#include <falcon/corefunc.h>
#include <falcon/corerange.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/membuf.h>
#include <falcon/garbagepointer.h>
#include <falcon/garbagelock.h>


#include <string>
#include <typeinfo>

#define GC_IDLE_TIME 250
// default 128k
#define GC_THREAD_STACK_SIZE  0x10000


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
   m_bLive(false),
   m_bRequestSweep( false ),
   m_lockGen( 0 )
{
   m_vmRing = 0;

   // use a ring for garbage items.
   m_garbageRoot = new GarbageableBase;
   m_garbageRoot->nextGarbage( m_garbageRoot );
   m_garbageRoot->prevGarbage( m_garbageRoot );

   // Use a ring also for the garbageLock system.
   m_lockRoot = new GarbageLock( true );
   m_lockRoot->next( m_lockRoot );
   m_lockRoot->prev( m_lockRoot );

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
   delete m_newRoot;
   delete m_garbageRoot;

   // delete the garbage lock ring.
   GarbageLock *ge = m_lockRoot->next();
   while( ge != m_lockRoot )
   {
      GarbageLock *gnext = ge->next();
      delete ge;
      ge = gnext;
   }
   delete ge;


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
   // already registered
   if( vm->m_nextVM != 0 || vm->m_prevVM )
      return;

   vm->m_idlePrev = vm->m_idleNext = 0;
   vm->incref();

   m_mtx_vms.lock();
   vm->m_generation = ++m_generation; // rollover detection in run()
   ++m_vmCount;

   if ( m_vmRing == 0 )
   {
      m_vmRing = vm;
      vm->m_nextVM = vm;
      vm->m_prevVM = vm;

      m_mingen = vm->m_generation;
      vm->incref();
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
         vm->incref();
         m_olderVM->decref();
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

   // ok, the VM is not held here anymore.
   vm->decref();
}


// WARNING -- this must be called with m_mtx_vms locked
void MemPool::electOlderVM()
{
   // Nay, we don't have any VM.
   if ( m_vmRing == 0 )
   {
      if ( m_olderVM != 0 )
      {
         m_olderVM->decref();
         m_olderVM = 0;
      }
   }
   else
   {
      VMachine *vmc = m_vmRing;
      m_mingen = vmc->m_generation;
      VMachine* vmmin = vmc;
      vmc = vmc->m_nextVM;

      while( vmc != m_vmRing )
      {
         if ( vmc->m_generation < m_mingen )
         {
            m_mingen = vmc->m_generation;
            vmmin = vmc;
         }
         vmc = vmc->m_nextVM;
      }
      vmmin->incref();
      if( m_olderVM != 0 )
         m_olderVM->decref();
      m_olderVM = vmmin;
   }
}


void MemPool::clearRing( GarbageableBase *ringRoot )
{
   TRACE( "Entering sweep %d, allocated %d", gcMemAllocated(), m_allocatedItems );
   // delete the garbage ring.
   int32 killed = 0;
   GarbageableBase *ring = m_garbageRoot->nextGarbage();

   // live modules must be killed after all their data. For this reason, we must put them aside.
   GarbageableBase *later_ring = 0;
   while( ring != m_garbageRoot )
   {
      if ( ring->mark() < m_mingen )
      {
         ring->nextGarbage()->prevGarbage( ring->prevGarbage() );
         ring->prevGarbage()->nextGarbage( ring->nextGarbage() );
         GarbageableBase *dropped = ring;
         ring = ring->nextGarbage();

         // a module? -- do it later
         if( ! dropped->finalize() )
         {
            dropped->nextGarbage(later_ring);
            dropped->prevGarbage( 0 );
            later_ring = dropped;
         }
         else
            killed++;
      }
      else {
         ring = ring->nextGarbage();
      }
   }

   TRACE( "Sweeping step 1 complete %d", gcMemAllocated() );

   // deleting persistent finalized items.
   while( later_ring != 0 )
   {
      GarbageableBase *current = later_ring;
      later_ring = later_ring->nextGarbage();
      delete current;
      killed++;
   }
   TRACE( "Sweeping step 2 complete %d", gcMemAllocated() );

   m_mtx_newitem.lock();
   fassert( killed <= m_allocatedItems );
   m_allocatedItems -= killed;
   m_mtx_newitem.unlock();

   TRACE( "Sweeping done, allocated %d (killed %d)", m_allocatedItems, killed );
}


void MemPool::storeForGarbage( Garbageable *ptr )
{
   // We mark newly created items as the maximum possible value
   // so they can't be reclaimed until marked at least once.
   ptr->mark( MAX_GENERATION );

   m_mtx_newitem.lock();
   m_allocatedItems++;

   ptr->nextGarbage( m_newRoot );
   ptr->prevGarbage( m_newRoot->prevGarbage() );
   m_newRoot->prevGarbage()->nextGarbage( ptr );
   m_newRoot->prevGarbage( ptr );
   m_mtx_newitem.unlock();
}

void MemPool::accountItems( int itemCount )
{
   m_mtx_newitem.lock();
   m_allocatedItems += itemCount;
   m_mtx_newitem.unlock();
}


bool MemPool::markVM( VMachine *vm )
{
   // mark all the messaging system.
   vm->markSlots( generation() );

   // mark the global symbols
   // When generational gc will be on, this won't be always needed.
   MapIterator iter = vm->liveModules().begin();
   while( iter.hasCurrent() )
   {
      LiveModule *currentMod = *(LiveModule **) iter.currentValue();
      // We must mark the current module.
      currentMod->gcMark( generation() );

      iter.next();
   }

   // mark all the items in the coroutines.
   ListElement *ctx_iter = vm->getCtxList()->begin();
   uint32 pos;
   while( ctx_iter != 0 )
   {
      VMContext *ctx = (VMContext *) ctx_iter->data();

      markItem( ctx->regA() );
      markItem( ctx->regB() );
      markItem( ctx->latch() );
      markItem( ctx->latcher() );
      markItem( ctx->self() );

      markItem( vm->regBind() );
      markItem( vm->regBindP() );

      StackFrame* sf = ctx->currentFrame();
      while( sf != 0 )
      {
         Item* stackItems = sf->stackItems();
         uint32 sl = sf->stackSize();
         markItem( sf->m_self );
         markItem( sf->m_binding );

         for( pos = 0; pos < sl; pos++ ) {
            markItem( stackItems[ pos ] );
         }
         sf = sf->prev();
      }

      ctx_iter = ctx_iter->next();
   }

   return true;
}

void MemPool::markItem( const Item &item )
{
   uint32 gen = generation();

   switch( item.type() )
   {
   case FLC_ITEM_RANGE:
      item.asRange()->gcMark( gen );
      break;

   case FLC_ITEM_GCPTR:
      item.asGCPointerShell()->gcMark( gen );
      break;

   case FLC_ITEM_ARRAY:
      item.asArray()->gcMark( gen );
      break;

   case FLC_ITEM_DICT:
      item.asDict()->gcMark( gen );
      break;

   case FLC_ITEM_OBJECT:
      item.asObject()->gcMark( gen );
      break;

   case FLC_ITEM_MEMBUF:
      item.asMemBuf()->gcMark( gen );
      break;

   case FLC_ITEM_CLASS:
      item.asClass()->gcMark( gen );
      break;

   case FLC_ITEM_FUNC:
      item.asFunction()->gcMark( gen );
      break;

   case FLC_ITEM_REFERENCE:
   {
      GarbageItem *gi = item.asReference();
      if( gi->mark() != gen ) {
         gi->mark( gen );
         markItem( gi->origin() );
      }
   }
   break;

   case FLC_ITEM_LBIND:
      if ( item.asFBind() != 0 )
      {
         item.asFBind()->gcMark( gen );
      }

      // fallthrough

   case FLC_ITEM_STRING:
      {
         String* str = item.asString();
         if( str->isCore() )
         {
            StringGarbage &gs = static_cast<CoreString*>(str)->garbage();
            gs.mark( gen );
         }

      }
      break;

      case FLC_ITEM_METHOD:
      {
         // if the item isn't alive, give it the death blow.
         if( item.asMethodFunc()->mark() != gen )
         {
            CallPoint* cp = item.asMethodFunc();
            cp->gcMark( gen );
         }

         Item self;
         item.getMethodItem( self );
         markItem( self );
      }
      break;

      case FLC_ITEM_CLSMETHOD:
      {
         CoreObject *co = item.asMethodClassOwner();
         if( co->mark() != gen ) {
            co->gcMark( gen );
         }

         CoreClass *cls = item.asMethodClass();
         // if the class is the generator of the method, we have already marked it.
         if( cls->mark() != gen )
         {
            cls->gcMark( gen );
         }
      }
      break;

      // all the others are shallow items; already marked
   }

}


void MemPool::gcSweep()
{
   TRACE( "Sweeping %d (mingen: %d, gen: %d)", gcMemAllocated(), m_mingen, m_generation );
   m_mtx_ramp.lock();
   RampMode *rm = m_curRampMode;
   if( m_curRampMode != 0 )
   {
      rm->onScanInit();
   }
   m_mtx_ramp.unlock();

   clearRing( m_garbageRoot );

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

void MemPool::performGC()
{
   m_mtxRequest.lock();
   m_bRequestSweep = true;
   m_eRequest.set();
   m_mtxRequest.unlock();

   m_eGCPerformed.wait();
}

//===================================================================================
// MT functions
//

void MemPool::idleVM( VMachine *vm, bool bPrio )
{
   // ok, we're givin the VM to the GC, so we reference it.
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
   vm->incref();

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
      m_th->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
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
      bMoreWork = false; // in case of sweep request, loop without pause.

      // first, detect the operating status.
      size_t memory = gcMemAllocated();
      int state = m_bRequestSweep || memory >= m_thresholdActive ? 2 :      // active mode
                  memory >= m_thresholdNormal ? 1 :      // normal mode
                  0;                                     // dormient mode

      TRACE( "Working %d (in mode %d)", gcMemAllocated(), state );

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
            MESSAGE( "Storing the garbage new ring in the normal ring" );
            newRingBack->nextGarbage( m_garbageRoot );
            newRingFront->prevGarbage( m_garbageRoot->prevGarbage() );
            m_garbageRoot->prevGarbage()->nextGarbage( newRingFront );
            m_garbageRoot->prevGarbage( newRingBack );
         }
         else
            m_mtx_newitem.unlock();
      }
      else {
         m_mtx_newitem.unlock();
         MESSAGE( "Skipping new ring inclusion due to safe area lock." );
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
               TRACE( "Activating blocking request vm %p", vm );
               vm->baton().block();
            }
            vm = vm->m_nextVM;
            while( vm != m_vmRing )
            {
               if ( vm->isGcEnabled() )
               {
                  TRACE( "Activating blocking request vm %p", vm );
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
            // we're done with this VM here
            vm->decref();
            m_mtx_idlevm.unlock();

            TRACE( "Discarding idle vm %p", vm );
            continue;
         }

         // mark the idle VM if we're not dormient.
         // ok, we need to reclaim some memory.
         // (try to acquire only if this is not a priority scan).
         if ( ! bPriority && ! vm->baton().tryAcquire() )
         {
            m_mtx_idlevm.unlock();
            // we're done with this VM here
            vm->decref();
            TRACE( "Was going to mark vm %p, but forfaited", vm );
            // oh damn, we lost the occasion. The VM is back alive.
            continue;
         }

         m_mtx_idlevm.unlock();

         m_mtx_vms.lock();
         // great; start mark loop -- first, set the new generation.
         advanceGeneration( vm, oldGeneration );
         m_mtx_vms.unlock();

         TRACE( "Marking idle vm %p at %d", vm, m_generation );

         // and then mark
         markVM( vm );
         // should notify now?
         if ( bPriority )
         {
            if ( ! vm->m_bWaitForCollect )
            {
               bPriority = false; // disable the rest.
               vm->m_eGCPerformed.set();
               vm->decref();
            }
         }
         else
         {
            // the VM is now free to go -- but it is not declared idle again.
            vm->baton().releaseNotIdle();
            // we're done with this VM here
            vm->decref();
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

               TRACE( "Marking oldest vm %p at %d", vm, m_generation );
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
      if ( oldMingen != m_mingen || bPriority || m_bRequestSweep )
      {
         bool signal = false;
         m_mtxRequest.lock();
         m_mtx_vms.lock();
         if ( m_bRequestSweep )
         {
            if ( m_vmCount == 0  )
            {
               // be sure to clear the garbage
               oldMingen = m_mingen;
               m_mingen = SWEEP_GENERATION;
               m_bRequestSweep = false;
               signal = true;
            }
            else {
               // HACK: we are not able to kill correctly VMS in multithreading in 0.9.1,
               // so we just let the request go;
               // we'll clean them during 0.9.1->0.9.2
               //m_bRequestSweep = true;
               //signal = true;
               TRACE( "Priority with %d", m_vmCount );
            }
         }
         m_mtx_vms.unlock();
         m_mtxRequest.unlock();

         // before sweeping, mark -- eventually -- the locked items.
         markLocked();

         // all is marked, we can sweep
         gcSweep();

         // should we notify about the sweep being complete?

         if ( bPriority )
         {
            fassert( vm != 0 );
            vm->m_eGCPerformed.set();
            vm->decref();
         }

         if ( signal )
         {
            m_mingen = oldMingen;
            m_eGCPerformed.set();
         }

         // no more use for this vm
      }

      oldGeneration = m_generation;  // spurious read is ok here (?)
      oldMingen = m_mingen;

      // if we have nothing to do, we shall wait a bit.
      if( ! bMoreWork )
      {
         MESSAGE( "Waiting GC idle time" );
         m_eRequest.wait(GC_IDLE_TIME);
      }
   }

   TRACE( "Stopping %d", gcMemAllocated() );
   return 0;
}


// to be called with m_mtx_vms locked
void MemPool::advanceGeneration( VMachine* vm, uint32 oldGeneration )
{
   uint32 curgen = ++m_generation;

   // detect here rollover.
   if ( curgen < oldGeneration || curgen >= MAX_GENERATION )
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

   m_vmRing->incref();
   if ( m_olderVM != 0 )
      m_olderVM->decref();

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


void MemPool::addGarbageLock( GarbageLock* ptr )
{
   m_mtx_lockitem.lock();
   ptr->next( m_lockRoot->next() );
   ptr->prev( m_lockRoot );
   m_lockRoot->next()->prev( ptr );
   m_lockRoot->next( ptr );
   m_mtx_lockitem.unlock();

   markItem( ptr->item() );
}


void MemPool::removeGarbageLock( GarbageLock *ptr )
{
   // frirst: remove the item from the availability pool
   m_mtx_lockitem.lock();
   ptr->next()->prev( ptr->prev() );
   ptr->prev()->next( ptr->next() );
   m_mtx_lockitem.unlock();
}


void MemPool::markLocked()
{
   fassert( m_lockRoot != 0 );

   // is there any VM keeping the locked items alive?
   if ( m_mingen <= m_lockGen )
      return;

   m_lockGen = m_generation;

   // Lock root never changes.
   GarbageLock *rlock = this->m_lockRoot;
   GarbageLock *lock = rlock;
   do
   {
      // The root item never needs to be marked
      m_mtx_lockitem.lock();
      lock = lock->next();
      m_mtx_lockitem.unlock();

      // if a new item is inserted now, NP:
      // it gets marked with current generation.
      // If it gets deleted, it will just get an extra mark
      // and live for an extra turn.
      memPool->markItem( lock->item() );

   } while( lock != rlock );
}

//=======================================================================
// Garbage Lock
//=======================================================================
GarbageLock::GarbageLock( bool )
{
}

GarbageLock::GarbageLock()
{
	memPool->addGarbageLock( this );
}

GarbageLock::GarbageLock( const Item &itm ):
	m_item(itm)
{
	memPool->addGarbageLock( this );
}

GarbageLock::~GarbageLock()
{
	memPool->removeGarbageLock( this );
}

}

/* end of mempool.cpp */
