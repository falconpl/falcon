/*
   FALCON - The Falcon Programming Language.
   FILE: collector.cpp

   Falcon Garbage Collector
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Jan 2011 13:16:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>

#include <falcon/memory.h>
#include <falcon/collector.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/gctoken.h>
#include <falcon/gclock.h>

#include <string>
#include <typeinfo>

#if FALCON_TRACE_GC
   #include <falcon/textwriter.h>
   #include <map>
#endif

#define GC_IDLE_TIME 250
// default 1M
#define GC_THREAD_STACK_SIZE  1024*1024

// By default, 1MB
#define TEMP_MEM_THRESHOLD 1000000

#define MAX_RECYCLE_LOCK_COUNT 1000
#define MAX_RECYCLE_TOKEN_COUNT 1000

namespace Falcon {

class Collector::Private
{
public:
#if FALCON_TRACE_GC
   typedef std::map<void*, Collector::DataStatus* > HistoryMap;
   HistoryMap m_hmap;
#endif

   Private() {}
   ~Private()
   {
      clearTrace();
   }

   void clearTrace()
   {
   #if FALCON_TRACE_GC
      HistoryMap::iterator hmi = m_hmap.begin();
      while( hmi != m_hmap.end() )
      {
         delete hmi->second;
         ++hmi;
      }
   #endif
   }
};

#include <time.h>

//================================================================
// History entry
//

Collector::Collector():
   m_mingen( 0 ),
   m_vmCount(0),
   m_vmIdle_head( 0 ),
   m_vmIdle_tail( 0 ),
   m_generation( 0 ),
   m_allocatedItems( 0 ),
   m_allocatedMem( 0 ),
   m_th(0),
   m_bLive(false),
   m_bRequestSweep( false ),
   m_bTrace( false ),
   _p( new Private )
{
   // use a ring for garbage items.
   m_garbageRoot = new GCToken(this, 0,0);
   m_garbageRoot->m_next = m_garbageRoot;
   m_garbageRoot->m_prev = m_garbageRoot;

   // separate the newly allocated items to allow allocations during sweeps.
   m_newRoot = new GCToken(this, 0,0);
   m_newRoot->m_next = m_newRoot;
   m_newRoot->m_prev = m_newRoot;

   // Use a spare area to minimize token creation
   m_recycleTokens = 0;
   m_recycleTokensCount = 0;

   // Use a ring also for the garbageLock system.
   m_lockRoot = new GCLock();
   m_lockRoot->m_next = m_lockRoot;
   m_lockRoot->m_prev = m_lockRoot;

   // Use a spare area to minimize token creation
   m_recycleLock = 0;
   m_recycleLockCount = 0;

   m_thresholdNormal = TEMP_MEM_THRESHOLD;
   m_thresholdActive = TEMP_MEM_THRESHOLD*3;

   // fill the ramp algorithms
   m_ramp[RAMP_MODE_OFF] = new RampNone;
   m_ramp[RAMP_MODE_STRICT_ID] = new RampStrict;
   m_ramp[RAMP_MODE_LOOSE_ID] = new RampLoose;
   m_ramp[RAMP_MODE_SMOOTH_SLOW_ID] = new RampSmooth( 2.6 );
   m_ramp[RAMP_MODE_SMOOTH_FAST_ID] = new RampSmooth( 6.5 );

   // force initialization in rampMode by setting a different initial value;
   m_curRampID = DEFAULT_RAMP_MODE+1;
   rampMode( DEFAULT_RAMP_MODE );
}


Collector::~Collector()
{
   // ensure the thread is down.
   stop();

   // Clear the rings, which will cause module unloading.
   clearRing( m_newRoot );
   clearRing( m_garbageRoot );

   delete m_newRoot;
   delete m_garbageRoot;

   // destroy the spare token elements
   GCToken* rectoken = m_recycleTokens;
   while( rectoken != 0 ) {
      GCToken* next = rectoken->m_next;
      delete rectoken;
      rectoken = next;
   }

   // clear the locks and the spares by opening the lock loop
   // -- first open the ring
   m_lockRoot->m_next->m_next = m_recycleLock;
   m_recycleLock = m_lockRoot->m_next;
   m_lockRoot->m_next = 0;

   // -- then delete all together.
   GCLock* litem = m_recycleLock;
   while( litem != 0 )
   {
      GCLock* next = litem->m_next;
      delete litem;
      litem = next;
   }

   // VMs are not mine, and they should be already dead since long.
   for( uint32 ri = 0; ri < RAMP_MODE_COUNT; ri++ )
      delete m_ramp[ri];

   delete _p;
}


bool Collector::rampMode( int mode )
{
   if( mode >= 0 && mode < RAMP_MODE_COUNT )
   {
      m_mtx_ramp.lock();
      if ( m_curRampID != mode )
      {
         m_curRampID = mode;
         m_curRampMode = m_ramp[mode];
         m_curRampMode->reset();
      }
      m_mtx_ramp.unlock();
      return true;
   }

   return false;
}


int Collector::rampMode() const
{
   m_mtx_ramp.lock();
   int mode = m_curRampID;
   m_mtx_ramp.unlock();
   return mode;
}



#if 0
void Collector::registerVM( VMachine *vm )
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


void Collector::unregisterVM( VMachine *vm )
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
void Collector::electOlderVM()
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

#endif

void Collector::clearRing( GCToken *ringRoot )
{
   //TRACE( "Entering sweep %ld, allocated %ld", (long)gcMemAllocated(), (long)m_allocatedItems );
   // delete the garbage ring.
   int32 killed = 0;
   GCToken *ring = ringRoot->m_next;

   // live modules must be killed after all their data. For this reason, we must put them aside.
   while( ring != ringRoot )
   {
      if ( ring->m_mark < m_mingen )
      {
         ring->m_next->m_prev =  ring->m_prev ;
         ring->m_prev->m_next =  ring->m_next ;
         GCToken *dropped = ring;
         ring = ring->m_next;

#if FALCON_TRACE_GC
         if( m_bTrace )
         {
            onDestroy( dropped->m_data );
         }
#endif
         dropped->m_cls->dispose(dropped->m_data);
         disposeToken( dropped );
         killed++;
      }
      else {
         ring = ring->m_next;
      }
   }

   //TRACE( "Sweeping step 1 complete %ld", (long)gcMemAllocated() );

   m_mtx_newitem.lock();
   fassert( killed <= m_allocatedItems );
   m_allocatedItems -= killed;
   m_mtx_newitem.unlock();

   TRACE( "Sweeping done, allocated %ld (killed %ld)", (long)m_allocatedItems, (long)killed );
}


GCToken* Collector::getToken( Class* cls, void* data )
{
   m_mtx_recycle_tokens.lock();
   // do we have a free token?
   if( m_recycleTokens != 0 )
   {
      GCToken* token = m_recycleTokens;
      m_recycleTokens = token->m_next;
      m_recycleTokensCount--;
      m_mtx_recycle_tokens.unlock();

      token->m_cls = cls;
      token->m_data = data;

      return token;
   }
   m_mtx_recycle_tokens.unlock();

   return new GCToken( this, cls, data );
}


void Collector::disposeToken(GCToken* token)
{
   m_mtx_recycle_tokens.lock();
   if ( m_recycleTokensCount > MAX_RECYCLE_TOKEN_COUNT )
   {
      m_mtx_recycle_tokens.unlock();
      delete token;
   }
   else
   {
      m_recycleTokensCount++;
      token->m_next = m_recycleTokens;
      m_recycleTokens = token;
      m_mtx_recycle_tokens.unlock();
   }
}


GCToken* Collector::store( const Class* cls, void *data )
{
   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );

   // put the element in the new list.
   m_mtx_newitem.lock();
   m_allocatedItems++;
   token->m_next =  m_newRoot ;
   token->m_prev =  m_newRoot->m_prev ;
   m_newRoot->m_prev->m_next =  token;
   m_newRoot->m_prev =  token;
   m_mtx_newitem.unlock();

   return token;
}


GCLock* Collector::storeLocked( const Class* cls, void *data )
{
   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );
   
   GCLock* l = this->lock( token );

   // put the element in the new list.
   m_mtx_newitem.lock();
   m_allocatedItems++;
   token->m_next =  m_newRoot ;
   token->m_prev =  m_newRoot->m_prev ;
   m_newRoot->m_prev->m_next =  token;
   m_newRoot->m_prev =  token;
   m_mtx_newitem.unlock();

   return l;
}


#if 0
bool Collector::markVM( VMachine *vm )
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

#endif


void Collector::gcSweep()
{

   //TRACE( "Sweeping %ld (mingen: %d, gen: %d)", (long)gcMemAllocated(), m_mingen, m_generation );

   m_mtx_ramp.lock();
   // ramp mode may change while we do the lock...
   RampMode* rm = m_curRampMode;
   rm->onScanInit();
   m_mtx_ramp.unlock();

   clearRing( m_garbageRoot );

   m_mtx_ramp.lock();
   rm->onScanComplete();
   m_thresholdActive = rm->activeLevel();
   m_thresholdNormal = rm->normalLevel();
   m_mtx_ramp.unlock();
}

int32 Collector::allocatedItems() const
{
   m_mtx_newitem.lock();
   int32 size = m_allocatedItems;
   m_mtx_newitem.unlock();

   return size;
}

void Collector::performGC()
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

void Collector::idleVM( VMachine *, bool )
{
#if 0
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
#endif

   m_mtx_idlevm.unlock();
   // wake up if we're waiting.
   m_eRequest.set();
}

void Collector::start()
{
   if ( m_th == 0 )
   {
      m_bLive = true;
      m_th = new SysThread( this );
      m_th->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
   }
}


void Collector::markNew()
{
   // NOTE: This method is called from the scan loop
   // which is the only thread authorized to increase m_generation;
   // In other words, the value of m_generation is constant in this method.

   MESSAGE( "Marking the items in the new ring as new." );
   bool bDone = false;

   m_mtx_newitem.lock();

   GCToken* newRingFront = m_newRoot->m_next;
   GCToken* newRingBack = 0;
   if ( newRingFront != m_newRoot )
   {
      bDone = true;
      newRingBack = m_newRoot->m_prev;
      // make the loop to turn into a list;
      newRingBack->m_next = 0;
      // disengage all the loop
      m_newRoot->m_next = m_newRoot;
      m_newRoot->m_prev = m_newRoot;
      m_mtx_newitem.unlock();
   }
   else
   {
      // nothing to do
      bDone = false;
      m_mtx_newitem.unlock();
   }

   if ( bDone )
   {
      uint32 mark = m_generation;
      MESSAGE( "Found items to be marked" );
      
      // first mark
      GCToken* newRing = newRingFront;
      while( newRing != 0 )
      {
         newRing->m_mark = mark;
         newRing->m_cls->gcMarkInstance( newRing->m_data, mark );
         newRing = newRing->m_next;
      }

      // then add to the standard garbage system
      // no need for locking; we're the only thread accessing here.
      m_garbageRoot->m_next->m_prev = newRingBack;
      newRingBack->m_next = m_garbageRoot->m_next;
      m_garbageRoot->m_next = newRingFront;
      newRingFront->m_prev = m_garbageRoot;
   }
   else
   {
      MESSAGE( "Skipping new ring inclusion." );
   }
}

void Collector::stop()
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

void* Collector::run()
{
   //uint32 oldGeneration = m_generation;
   //uint32 oldMingen = m_mingen;
   bool bMoreWork;

   while( m_bLive )
   {
      bMoreWork = false; // in case of sweep request, loop without pause.

#if 0
      // first, detect the operating status.
      size_t memory = gcMemAllocated();
      int state = m_bRequestSweep || memory >= m_thresholdActive ? 2 :      // active mode
                  memory >= m_thresholdNormal ? 1 :      // normal mode
                  0;                                     // dormient mode

      TRACE( "Working %ld (in mode %d)", (long)gcMemAllocated(), state );

      

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
#endif
      
      // if we have nothing to do, we shall wait a bit.
      if( ! bMoreWork )
      {
         MESSAGE( "Waiting GC idle time" );
         m_eRequest.wait(GC_IDLE_TIME);
      }
   }

   //TRACE( "Stopping %ld", (long)gcMemAllocated() );
   return 0;
}


// to be called with m_mtx_vms locked
void Collector::advanceGeneration( VMachine*, uint32 oldGeneration )
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

#if 0
   vm->m_generation = curgen;

   // Now that we have marked it, if this was the oldest VM, we need to elect the new oldest vm.
   if ( vm == m_olderVM )
   {
      // calling it with mtx_vms locked
      electOlderVM();
   }
#endif
}


void Collector::remark( uint32 curgen )
{
   GCToken *ring = m_garbageRoot->m_next;

   // live modules must be killed after all their data. For this reason, we must put them aside.
   while( ring != m_garbageRoot )
   {
      // Don't mark objects that are still unassigned.
      if( ring->m_mark != MAX_GENERATION )
         ring->m_mark = curgen;

      ring = ring->m_next;
   }

}


// WARNING: Rollover is to be called with m_mtx_vms locked.
void Collector::rollover()
{
   // Sets the minimal VM.
   m_mingen = 1;
#if 0
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
#endif
}



void Collector::markLocked()
{
   fassert( m_lockRoot != 0 );
   
   // temporarily disconnect the lock loop structure
   m_mtx_lockitem.lock();
   GCLock *firstLock = this->m_lockRoot->m_next;
   GCLock *lastLock = this->m_lockRoot->m_prev;
   if( firstLock != lastLock )
   {
      m_lockRoot->m_next = m_lockRoot;
      m_lockRoot->m_prev = m_lockRoot;
      lastLock->m_next = 0;
      m_mtx_lockitem.unlock();
   }
   else
   {
      // nothing to mark
      m_mtx_lockitem.unlock();
      return;
   }

   // mark the items
   uint32 mark = m_generation;
   GCLock *lock = firstLock;
   while( lock != 0 )
   {
      // no need to mark this?
      if( lock->m_bDisposed )
      {
         // advance the head in case we're pointing to it
         if ( firstLock == lock )
         {
            firstLock = lock->m_next;
         }
         
         GCLock* next = lock;
         this->disposeLock(lock);
         lock = next;
      }
      else
      {
         const Item& item = lock->item();
         item.gcMark( mark );
         
         lock = lock->m_next;
      }
   }

   // re-add the locks that are left
   if ( firstLock != 0 )
   {
      m_mtx_lockitem.lock();
      lastLock->m_next = m_lockRoot->m_next;
      m_lockRoot->m_next->m_prev = lastLock;
      firstLock->m_prev = m_lockRoot;
      m_lockRoot->m_next = firstLock;
      m_mtx_lockitem.unlock();
   }

   // we don't care about locks added in the meanwhile, as they belong to the
   // new ring, and cannot be reaped till the next mark/sweep loop
}


void Collector::disposeLock( GCLock* lock )
{
   m_mtx_recycle_locks.lock();
   if ( m_recycleLockCount > MAX_RECYCLE_LOCK_COUNT )
   {
      m_mtx_recycle_locks.unlock();
      delete lock;
   }
   else
   {  
      m_recycleLockCount++;
      lock->m_next = m_recycleLock;
      m_recycleLock = lock;
      m_mtx_recycle_locks.unlock();
   }
}


GCLock* Collector::lock( const Item& item )
{
   GCLock* l = 0;
   
   m_mtx_recycle_locks.lock();
   // do we have a free token?
   if( m_recycleLock != 0 )
   {
      GCLock* l = m_recycleLock;
      m_recycleLock = l->m_next;
      m_recycleLockCount--;
      m_mtx_recycle_locks.unlock();

      l->m_bDisposed = false;
      l->m_item = item;
   }
   else
   {
      m_mtx_recycle_locks.unlock();
      l = new GCLock( item );
   }

   // save the item.
   m_mtx_lockitem.lock();
   m_lockRoot->m_next->m_prev = l;
   l->m_next = m_lockRoot->m_next;
   m_lockRoot->m_next = l;
   l->m_prev = m_lockRoot;
   m_lockRoot = l;
   m_mtx_lockitem.unlock();

   return l;
}

/** Unlocks a locked item. */
void Collector::unlock( GCLock* lock )
{
   m_mtx_lockitem.lock();
   lock->m_next->m_prev = lock->m_prev;
   lock->m_prev->m_next = lock->m_next;
   m_mtx_lockitem.unlock();

   disposeLock( lock );
}

#if FALCON_TRACE_GC

GCToken* Collector::H_store( const Class* cls, void *data, const String& fname, int line )
{
   GCToken* token = store( cls, data );
   if ( m_bTrace )
   {
      onCreate( cls, data, fname, line );
   }
   
   return token;
}

GCLock* Collector::H_storeLocked( const Class* cls, void *data, const String& file, int line )
{
   GCLock* lock = storeLocked( cls, data );
   if ( m_bTrace )
   {
      onCreate( cls, data, file, line );
   }
   return lock;
}


bool Collector::trace() const
{
   m_mtx_history.lock();
   bool bTrace = m_bTrace;
   m_mtx_history.unlock();
   return bTrace;
}


void Collector::trace( bool t )
{ 
   m_mtx_history.lock();
   m_bTrace = t;
   m_bTraceMarks = t;
   m_mtx_history.unlock();
}


bool Collector::traceMark() const
{
   m_mtx_history.lock();
   bool bTrace = m_bTraceMarks;
   m_mtx_history.unlock();
   return bTrace;
}

void Collector::traceMark( bool t )
{
   m_mtx_history.lock();
   m_bTraceMarks = t;
   m_mtx_history.unlock();
}


void Collector::onCreate( const Class* cls, void* data, const String& file, int line )
{
   m_mtx_history.lock();
   Private::HistoryMap::iterator iter = _p->m_hmap.find( data );
   if( iter != _p->m_hmap.end() )
   {
      DataStatus& status = *iter->second;
      if( status.m_bAlive )
      {
         m_mtx_history.unlock();
         // ops, we lost the previous item.
         String s;
         s.A("While creating a GC token for 0x").H( (int64) data, true, 16)
         .A(" of class ").A( cls->name() ).A( " from ").A( file ).A(":").N(line).A("\n");
         s += "The item was already alive -- and we didn't reclaim it:\n";
         s += status.dump();

         Engine::die( s );
      }
      else
      {
         status.m_bAlive = true;
         status.addEntry( new HECreate( file, line, cls->name() ) );
         m_mtx_history.unlock();
      }
   }
   else
   {
      DataStatus* status = new DataStatus(data);
      _p->m_hmap[data] = status;
      status->addEntry( new HECreate( file, line, cls->name() ) );
      m_mtx_history.unlock();
   }
}


void Collector::onMark( void* data )
{
   m_mtx_history.lock();
   Private::HistoryMap::iterator iter = _p->m_hmap.find( data );
   if( iter != _p->m_hmap.end() )
   {
      DataStatus& status = *iter->second;
      if( status.m_bAlive )
      {
         // record this check only if really required
         if( m_bTraceMarks )
         {
            status.addEntry( new HEMark() );
         }
         m_mtx_history.unlock();
      }
      else
      {
         // ops, we lost the previous item.
         String s;
         s.A("While marking a GC token for 0x").H( (int64) data, true, 16);
         s += "The item is not alive -- crash ahead:\n";
         s += status.dump();

         Engine::die( s );
      }
   }
   else
   {
      m_mtx_history.unlock();
      // ops, we don't know about this item -- it might have been decalred before trace.
   }
}

void Collector::onDestroy( void* data )
{
   m_mtx_history.lock();
   Private::HistoryMap::iterator iter = _p->m_hmap.find( data );
   if( iter != _p->m_hmap.end() )
   {
      DataStatus& status = *iter->second;
      if( status.m_bAlive )
      {
         status.addEntry( new HEDestroy() );
         status.m_bAlive = false;
         m_mtx_history.unlock();
      }
      else
      {
         // ops, we lost the previous item.
         String s;
         s.A("While destroying a GC token for 0x").H( (int64) data, true, 16).A("\n");
         s += "The item is not alive -- crash ahead:\n";
         s += status.dump();

         Engine::die( s );
      }
   }
   else
   {
      m_mtx_history.unlock();
      // ops, we don't know about this item -- it might have been decalred before trace.
   }
}


void Collector::dumpHistory( TextWriter* target ) const
{
   Private::HistoryMap::iterator iter = _p->m_hmap.begin();
   while( iter != _p->m_hmap.end() )
   {
      target->writeLine( iter->second->dump() );
      ++iter;
   }
}

void Collector::enumerateHistory( DataStatusEnumerator& r ) const
{

   Private::HistoryMap::iterator iter = _p->m_hmap.begin();
   while( iter != _p->m_hmap.end() )
   {
      DataStatus& status = *iter->second;
      if( ! r( status, (++iter == _p->m_hmap.end() ) ) )
      {
         break;
      }
   }
}


Collector::DataStatus* Collector::getHistory( const void* pointer ) const
{
   Private::HistoryMap::const_iterator iter = _p->m_hmap.find( (void*)pointer );
   if( iter != _p->m_hmap.end() ) return iter->second;
   return 0;
}

void Collector::clearTrace()
{
   _p->clearTrace();
}

#endif


}

/* end of collector.cpp */


