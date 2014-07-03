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
#undef SRC
#define SRC "/engine/collector.cpp"

#include <falcon/trace.h>
#include <falcon/collector.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/gctoken.h>
#include <falcon/gclock.h>
#include <falcon/vmcontext.h>
#include <falcon/collectoralgorithm.h>
#include <falcon/shared.h>
#include <falcon/sys.h>
#include <falcon/poolable.h>
#include <falcon/modspace.h>

#include <falcon/log.h>

#include <stdio.h>

#include <deque>
#include <string>
#include <typeinfo>
#include <set>
#include <map>

#if FALCON_TRACE_GC
   #include <falcon/stdstreams.h>
   #include <falcon/textwriter.h>
#endif

#define GC_IDLE_TIME 250
// default 1M
#define GC_THREAD_STACK_SIZE  1024*1024

// By default, 1MB
#define TEMP_MEM_THRESHOLD 1000000

#define MAX_RECYCLE_LOCK_COUNT 1000
#define MAX_RECYCLE_TOKEN_COUNT 1000

namespace Falcon {

class Collector::Cmd: public Poolable
{
public:
   typedef enum {
      e_cmd_none,
      e_cmd_register,
      e_cmd_unregister,
      e_cmd_mark,
      e_cmd_fullgc,
      e_cmd_check,
      e_cmd_offer,
      e_cmd_terminate,
      e_cmd_abort,

      e_cmd_enable,
      e_cmd_disable,

      // internal messages
      e_start_mark,
      e_start_sweep,
      e_sweep_complete
   }
   t_type;

   t_type m_type;
   VMContext* m_ctx;
   Event* m_toBeSignaled;
   Shared* m_sharedToBeSignaled;

   typedef void (*callback )(void* data);
   callback m_cb;
   void* m_cbData;

   Cmd():
      m_type(e_cmd_none),
      m_ctx(0),
      m_toBeSignaled(0),
      m_sharedToBeSignaled(0),
      m_cb(0),
      m_cbData(0)
   {
   }


   Cmd( t_type t, VMContext* ctx=0, Event* evt=0, Shared* sh=0 ):
      m_type(t),
      m_ctx(ctx),
      m_toBeSignaled(evt),
      m_sharedToBeSignaled(sh),
      m_cb(0),
      m_cbData(0)
   {
      if( sh != 0 )
      {
         sh->incref();
      }

      if( ctx != 0 )
      {
         ctx->incref();
      }
   }

   Cmd(const Cmd& other):
      m_type( other.m_type ),
      m_ctx( other.m_ctx ),
      m_toBeSignaled( other.m_toBeSignaled ),
      m_sharedToBeSignaled( other.m_sharedToBeSignaled ),
      m_cb(other.m_cb),
      m_cbData(other.m_cbData)
   {
      if( m_sharedToBeSignaled != 0 )
      {
         m_sharedToBeSignaled->incref();
      }

      if( m_ctx != 0 )
      {
         m_ctx->incref();
      }
   }

   inline virtual ~Cmd() {
      clear();
   }


   inline void set( t_type t, VMContext* ctx=0, Event* evt=0, Shared* sh=0 )
   {
      m_type = t;
      m_ctx = ctx;
      m_toBeSignaled = evt;
      m_sharedToBeSignaled = sh;

      if( sh != 0 )
      {
         sh->incref();
      }

      if( ctx != 0 )
      {
         ctx->incref();
      }
   }

   inline void setCallback( callback cb, void* data )
   {
      m_cb = cb;
      m_cbData = data;
   }

   inline void clear()
   {
      if( m_sharedToBeSignaled != 0 )
      {
         m_sharedToBeSignaled->decref();
         m_sharedToBeSignaled = 0;
      }

      if( m_ctx != 0 )
      {
         m_ctx->decref();
         m_ctx = 0;
      }

      m_cb = 0;
      m_cbData = 0;
   }

   inline virtual void vdispose() { clear(); dispose(); }

   void signal()
   {
      if( m_cb != 0 )
      {
         TRACE1("Cmd::signal -- invoking callback %p(%p)", m_cb, m_cbData );
         m_cb(m_cbData);
      }

      if( m_toBeSignaled != 0 )
      {
         TRACE1("Cmd::signal -- signaling event %p", m_sharedToBeSignaled );
         m_toBeSignaled->set();
      }

      if( m_sharedToBeSignaled != 0 )
      {
         TRACE1("Cmd::signal -- signaling resource %p", m_toBeSignaled );
         m_sharedToBeSignaled->signal();
     }
   }

   inline bool isSignalable() const {
      return m_cb != 0 || m_toBeSignaled != 0 || m_sharedToBeSignaled != 0;
   }
};


class Collector::Private
{
public:
   // This mutex is used for all the context modify operations,
   // - m_contexts
   //
   Mutex m_mtx_contexts;

   typedef std::set<VMContext*> ContextSet;
   ContextSet m_contexts;
   ContextSet m_inspectedContexts;

   Mutex m_mtx_markingList;
   typedef std::deque<VMContext*> MarkingList;
   MarkingList m_markingList;

   Pool m_cmd_pool;

   // maker command queue
   Event m_markerWork;
   PoolFIFO m_markCommands;
   PoolFIFO m_markDelayed;

   // Sweeper command queue
   Event m_sweeperWork;
   PoolFIFO m_sweepCommands;

   // waiters saved in waiting to know if a mark loop has been performed
   PoolFIFO m_markWaiters;
   // waiters saved in waiting to know if a sweep loop has been performed
   PoolFIFO m_sweepWaiters;

#if FALCON_TRACE_GC
   typedef std::map<void*, Collector::DataStatus* > HistoryMap;
   HistoryMap m_hmap;
#endif

   Private() {}
   ~Private()
   {
      clearTrace();
   }

   void clearQueue(PoolFIFO& listeners)
   {
      while( ! listeners.empty() )
      {
         Cmd* cmd = listeners.tdeq<Cmd>();
         cmd->signal();
         cmd->vdispose();
      }
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

   void sendMarkMessage(Cmd::t_type t, VMContext* ctx=0, Event* evt = 0, Shared* sh = 0, Cmd::callback cb = 0, void* cbd = 0)
   {
      Cmd* cmd = m_cmd_pool.xget<Cmd>();
      cmd->set(t,ctx,evt,sh);
      cmd->setCallback(cb,cbd);
      m_markCommands.enqueue(cmd);
      m_markerWork.set();
   }

   void sendSweepMessage(Cmd::t_type t, VMContext* ctx=0, Event* evt = 0, Shared* sh = 0, Cmd::callback cb = 0, void* cbd = 0)
   {
      Cmd* cmd = m_cmd_pool.xget<Cmd>();
      cmd->set(t,ctx,evt,sh);
      cmd->setCallback(cb,cbd);
      m_sweepCommands.enqueue(cmd);
      m_sweeperWork.set();
   }
};

#include <time.h>

//================================================================
// History entry
//

Collector::Collector():
   m_thMarker(0),
   m_marker(this),
   m_thTimer(0),
   m_timer(this),
   m_thSweeper(0),
   m_sweeper(this),

   m_aLive(1),
   m_bTrace( false ),
   m_currentMark(0),
   m_storedMem(0),
   m_storedItems(0),
   m_status(e_status_green),
   m_markLoops(0),
   m_sweepLoops(0),
   _p( new Private )
{
   // start as enabled
   m_bEnabled = true;

   // use a ring for garbage items.
   m_garbageRoot = new GCToken(0,0);
   m_garbageRoot->m_next = m_garbageRoot;
   m_garbageRoot->m_prev = m_garbageRoot;

   // separate the newly allocated items to allow allocations during sweeps.
   m_newRoot = new GCToken(0,0);
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

   // fill the ramp algorithms
   m_algo  = new CollectorAlgorithm*[ FALCON_COLLECTOR_ALGORITHM_COUNT ];
   m_algo[FALCON_COLLECTOR_ALGORITHM_MANUAL] = new CollectorAlgorithmManual;
   m_algo[FALCON_COLLECTOR_ALGORITHM_FIXED] = new CollectorAlgorithmFixed(1000000);
   m_algo[FALCON_COLLECTOR_ALGORITHM_STRICT] = new CollectorAlgorithmStrict;
   m_algo[FALCON_COLLECTOR_ALGORITHM_SMOOTH] = new CollectorAlgorithmSmooth;
   m_algo[FALCON_COLLECTOR_ALGORITHM_LOOSE] = new CollectorAlgorithmLoose;

   m_curAlgoID = -1;
   setAlgorithm( FALCON_COLLECTOR_ALGORITHM_DEFAULT );
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

   for( uint32 ri = 0; ri < FALCON_COLLECTOR_ALGORITHM_COUNT; ri++ )
   {
      delete m_algo[ri];
   }
   delete _p;
}


//=================================================================================
// Interface
//


void Collector::enable( bool mode )
{
   // no need for locking, the marker can handle multiple messages.
   if( mode != m_bEnabled )
   {
      m_bEnabled = mode;
      _p->sendMarkMessage( mode ? Cmd::e_cmd_enable : Cmd::e_cmd_disable );
   }
}

bool Collector::isEnabled() const
{
   return m_bEnabled;
}

bool Collector::setAlgorithm( int mode )
{
   if( mode >= 0 && mode < FALCON_COLLECTOR_ALGORITHM_COUNT )
   {
      CollectorAlgorithm* algo = 0;

      m_mtx_algo.lock();
      if ( m_curAlgoID != mode )
      {
         m_curAlgoID = mode;
         m_curAlgoMode = m_algo[mode];
         algo = m_curAlgoMode;

         // reset thresholds
         m_memoryThreshold = (uint64)-1;
         m_itemThreshold = (uint64)-1;
      }
      m_mtx_algo.unlock();

      if( algo != 0 )
      {
         algo->onApply(this);
      }

      return true;
   }

   return false;
}



int Collector::currentAlgorithm() const
{
   m_mtx_algo.lock();
   int mode = m_curAlgoID;
   m_mtx_algo.unlock();
   return mode;
}

CollectorAlgorithm* Collector::currentAlgorithmObject() const
{
   m_mtx_algo.lock();
   CollectorAlgorithm* mode = m_curAlgoMode;
   m_mtx_algo.unlock();
   return mode;
}

void Collector::memoryThreshold( uint64 th, bool doNow )
{

   bool perform = false;
   m_mtx_accountmem.lock();
   int64 sm = m_storedMem;
   if(  th <= (uint64) m_storedMem )
   {
      th = -1;
      perform = true;
   }
   m_mtx_accountmem.unlock();

   if( doNow && perform )
   {
      currentAlgorithmObject()->onMemoryThreshold(this, sm );
   }

   m_mtx_algo.lock();
   m_memoryThreshold = th;
   m_mtx_algo.unlock();
}

void Collector::itemThreshold( uint64 th, bool doNow )
{
   bool perform;
   m_mtx_accountmem.lock();
   int64 sm = m_storedItems;
   if(  th <= (uint64) m_storedItems )
   {
      th = -1;
      perform = true;
   }
   m_mtx_accountmem.unlock();

   if( doNow && perform )
   {
      currentAlgorithmObject()->onItemThreshold(this, sm );
   }

   m_mtx_algo.lock();
   m_itemThreshold = th;
   m_mtx_algo.unlock();
}


void Collector::registerContext( VMContext *ctx, Event* evt )
{
   TRACE( "Collector::registerContext - %p(%d) in Process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );

   _p->sendMarkMessage(Cmd::e_cmd_register, ctx, evt);
}


void Collector::unregisterContext( VMContext *ctx )
{
   TRACE( "Collector::unregisterContext - %p(%d) in Process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );

   _p->sendMarkMessage(Cmd::e_cmd_unregister, ctx);
}


void Collector::offerContext( VMContext* ctx )
{
   TRACE( "Collector::offerContext -- being offered ctx %d(%p) in process %d(%p)",
            ctx->id(), ctx, ctx->process()->id(), ctx->process() );

   _p->sendMarkMessage(Cmd::e_cmd_offer, ctx);
}


void Collector::performGC( bool wait )
{
   Event markEvt;
   Cmd* cmd = _p->m_cmd_pool.xget<Cmd>();
   cmd->set(Cmd::e_cmd_fullgc);
   if(wait)
   {
      cmd->m_toBeSignaled = &markEvt;
   }

   _p->m_markCommands.enqueue(cmd);
   _p->m_markerWork.set();

   if( wait )
   {
      markEvt.wait();
   }
}


void Collector::performGCOnShared( Shared* shared )
{
   TRACE( "Collector::performGCOnShared -- %p", shared );
   Cmd* cmd = _p->m_cmd_pool.xget<Cmd>();
   cmd->set(Cmd::e_cmd_fullgc);

   cmd->m_sharedToBeSignaled = shared;
   shared->incref();

   _p->m_markCommands.enqueue(cmd);
   _p->m_markerWork.set();
}


void Collector::suggestGC()
{
   MESSAGE( "Collector::suggestGC" );
   Cmd* cmd = _p->m_cmd_pool.xget<Cmd>();
   cmd->set(Cmd::e_cmd_check);

   _p->m_markCommands.enqueue(cmd);
   _p->m_markerWork.set();
}

//=================================================================================
// Utilities
//

void Collector::clearRing( GCToken *ringRoot )
{
   TRACE( "Collector::clearRing %p %ld items, %ld bytes", ringRoot, (long)storedItems(), (long)storedMemory() );

   // delete the garbage ring.
   GCToken *ring = ringRoot->m_next;
   if( ring == ringRoot ) {
      MESSAGE("Collector::clearRing -- nothing to clear");
      return;
   }
   ringRoot->m_prev->m_next = 0;
   ringRoot->m_prev = ringRoot;
   ringRoot->m_next = ringRoot;

   int64 count = 0;
   int64 mem = 0;
   int32 prio = 0;

   GCToken* prioBegin = 0;
   GCToken* prioNow = 0;

   // live modules must be killed after all their data. For this reason, we must put them aside.
   while(true)
   {
      TRACE( "Collector::clearRing -- priority %d", prio );

      while( ring != 0 )
      {
   #if FALCON_TRACE_GC
         if( m_bTrace )
         {
            onDestroy( ring->m_data );
         }
   #endif

         Class* cls = ring->m_cls;

         if( cls->clearPriority() <= prio )
         {
            void* data = ring->m_data;
            mem += cls->occupiedMemory(data);
            cls->dispose(data);
            ++count;
            GCToken* prev = ring;
            ring = ring->m_next;
            delete prev;
         }
         else {
            // skip it now, and keep it for later.
            if( prioBegin == 0 ) {
               prioBegin = ring;
               prioNow = ring;
            }
            else {
               prioNow->m_next = ring;
               prioNow = ring;
            }
            ring = ring->m_next;
            prioNow->m_next = 0;
         }
      }

      if( prioNow == 0 ) {
         break;
      }

      ring= prioBegin;
      prioBegin = prioNow = 0;
      prio++;
      TRACE( "Collector::clearRing -- found more things to collect at higher prio %d", prio );
   }

   m_mtx_accountmem.lock();
   m_storedItems -= count;
   m_storedMem -= mem;
   m_mtx_accountmem.unlock();

   TRACE( "Collector::clearRing -- %p cleared %ld items, %ld bytes, remaining %ld items, %ld bytes",
            ringRoot, (long) count, (long)mem,
            (long)storedItems(), (long)storedMemory() );
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

   return new GCToken( cls, data );
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
   TRACE2( "Collector::store instance of %s: %p", cls->name().c_ize(), data);

#ifndef NDEBUG
   TRACE2( "Collector::store_in generic instance of %s: %p",
            cls->name().c_ize(), data);
#endif
   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );
   store_internal(cls, data, token);
   return token;
}


GCLock* Collector::storeLocked( const Class* cls, void *data )
{
   TRACE2( "Collector::storeLocked instance of %s: %p", cls->name().c_ize(), data);
   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );
   GCLock* l = this->lock( token );
   store_internal(cls, data, token);

   return l;
}

void Collector::store_internal( const Class* cls, void* data, GCToken* token )
{
   int64 memory = cls->occupiedMemory(data);
   m_mtx_accountmem.lock();
   uint64 stoi = (uint64) (++m_storedItems);
   uint64 stom = (uint64) (m_storedMem+= memory);
   m_mtx_accountmem.unlock();

   // we do without lock, not urgent
   if( stoi >= m_itemThreshold )
   {
      m_mtx_algo.lock();
      m_itemThreshold = (uint64) -1;
      CollectorAlgorithm* algo = m_curAlgoMode;
      m_mtx_algo.unlock();
      algo->onItemThreshold(this, stoi);
   }

   if( stom >= m_memoryThreshold )
   {
      m_mtx_algo.lock();
      m_memoryThreshold = (uint64) -1;
      CollectorAlgorithm* algo = m_curAlgoMode;
      m_mtx_algo.unlock();
      algo->onMemoryThreshold(this, stom);
   }

   // put the element in the new list.
   m_mtx_newRoot.lock();
   token->m_next =  m_newRoot->m_next;
   token->m_prev =  m_newRoot;
   m_newRoot->m_next->m_prev =  token;
   m_newRoot->m_next =  token;
   m_mtx_newRoot.unlock();
}

//===================================================================================
// MT functions
//


void Collector::start()
{
   MESSAGE( "Collector::start" );

   if ( m_thMarker == 0 )
   {
      atomicSet(m_aLive, 1);
      m_thMarker = new SysThread( &m_marker );
      m_thMarker->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
      m_thTimer = new SysThread( &m_timer );
      m_thTimer->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
      m_thSweeper = new SysThread( &m_sweeper );
      m_thSweeper->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
   }
}


void Collector::stop()
{
   MESSAGE( "Collector::stop" );

   if ( m_thMarker != 0 )
   {
      atomicSet(m_aLive,0);
      m_timerWork.set();

      // wake up our threads
      _p->sendMarkMessage(Cmd::e_cmd_terminate);
      _p->sendSweepMessage(Cmd::e_cmd_terminate);

      // join them
      void *dummy = 0;
      m_thMarker->join( dummy );
      m_thSweeper->join( dummy );
      m_thTimer->join( dummy );

      // and clear them
      m_thMarker = 0;
      m_thSweeper = 0;
      m_thTimer = 0;
   }
}


void Collector::algoTimeout( uint32 to )
{
   int64 now = to == 0 ? -1 : Sys::_milliseconds();

   m_mtx_timer.lock();
   m_algoRandezVous = to + now;
   m_mtx_timer.unlock();
   m_timerWork.set();
}


void Collector::markLocked( uint32 mark )
{
   fassert( m_lockRoot != 0 );

   // temporarily disconnect the lock loop structure
   m_mtx_lockitem.lock();
   GCLock *firstLock = this->m_lockRoot->m_next;
   GCLock *lastLock = m_lockRoot->m_prev;
   if( firstLock != m_lockRoot )
   {
      // reclose the ring.
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

   // mark the extracted items items
   GCLock *lock = firstLock;
   while( lock != 0 )
   {
      // no need to mark this?
      if( lock->m_bDisposed )
      {
         GCLock* next = lock->m_next;

         // disengage the current lock
         // advance the head/tail in case we're pointing to it
         if ( firstLock == lock )
         {
            firstLock = lock->m_next;
         }
         else
         {
            lock->m_prev->m_next = lock->m_next;
         }

         if( lastLock == lock )
         {
            lastLock = lock->m_prev;
         }
         else
         {
            lock->m_next->m_prev = lock->m_prev;
         }

         // send this lock to the memory pool
         this->disposeLock(lock);
         lock = next;
      }
      else
      {
         const Item* item = lock->itemPtr();
         Class* cls;
         void* data;
         if( item->asClassInst(cls, data ) ){
#if FALCON_TRACE_GC
            onMark(data);
#endif
            cls->gcMarkInstance(data, mark);
         }

         lock = lock->m_next;
      }
   }

   // re-add the locks that are left (if any)
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
   // new items ring, and cannot be reaped till the next mark/sweep loop
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
      l = m_recycleLock;
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
   l->m_ptrItem = &l->m_item;

   // save the item.
   m_mtx_lockitem.lock();
   m_lockRoot->m_next->m_prev = l;
   l->m_next = m_lockRoot->m_next;
   m_lockRoot->m_next = l;
   l->m_prev = m_lockRoot;
   m_mtx_lockitem.unlock();

   return l;
}


GCLock* Collector::lockPtr( Item* ptr )
{
   GCLock* l = 0;

   m_mtx_recycle_locks.lock();
   // do we have a free token?
   if( m_recycleLock != 0 )
   {
      l = m_recycleLock;
      m_recycleLock = l->m_next;
      m_recycleLockCount--;
      m_mtx_recycle_locks.unlock();

      l->m_bDisposed = false;
      l->m_ptrItem = ptr;
   }
   else
   {
      m_mtx_recycle_locks.unlock();
      l = new GCLock( Item() );
      l->m_ptrItem = ptr;
   }

   // save the item.
   m_mtx_lockitem.lock();
   m_lockRoot->m_next->m_prev = l;
   l->m_next = m_lockRoot->m_next;
   m_lockRoot->m_next = l;
   l->m_prev = m_lockRoot;
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

void Collector::accountMemory( int64 memory )
{
   m_mtx_accountmem.lock();
   int64 stom = m_storedMem += memory;
   m_mtx_accountmem.unlock();

   if( ((uint64)stom) >= m_memoryThreshold )
   {
      m_mtx_algo.lock();
      m_memoryThreshold = (uint64) -1;
      CollectorAlgorithm* algo = m_curAlgoMode;
      m_mtx_algo.unlock();
      algo->onMemoryThreshold(this, stom);
   }
}

int64 Collector::storedMemory() const
{
   m_mtx_accountmem.lock();
   int64 result = m_storedMem;
   m_mtx_accountmem.unlock();

   return result;
}

int64 Collector::storedItems() const
{
   m_mtx_accountmem.lock();
   int64 result = m_storedItems;
   m_mtx_accountmem.unlock();

   return result;
}


int64 Collector::activeItems() const
{
   m_mtx_accountmem.lock();
   int64 result = m_aliveItems;
   m_mtx_accountmem.unlock();

   return result;
}


int64 Collector::activeMemory() const
{
   m_mtx_accountmem.lock();
   int64 result = m_aliveMem;
   m_mtx_accountmem.unlock();

   return result;
}

void Collector::active(int64& mem, int64& items) const
{
   m_mtx_accountmem.lock();
   items = m_aliveItems;
   mem = m_aliveMem;
   m_mtx_accountmem.unlock();
}

int64 Collector::sweepLoops( bool clear ) const
{
   m_mtx_accountmem.lock();
   int64 result = m_sweepLoops;
   if( clear )
   {
      m_sweepLoops = 0;
   }
   m_mtx_accountmem.unlock();

   return result;
}

int64 Collector::markLoops( bool clear ) const
{
   m_mtx_accountmem.lock();
   int64 result = m_markLoops;
   if( clear )
   {
      m_markLoops = 0;
   }
   m_mtx_accountmem.unlock();

   return result;
}


void Collector::stored( int64& memory, int64& items ) const
{
   m_mtx_accountmem.lock();
   memory = m_storedMem;
   items = m_storedItems;
   m_mtx_accountmem.unlock();
}

void Collector::onSweepBegin()
{
  m_mtx_algo.lock();
  CollectorAlgorithm* algo = m_algo[m_curAlgoID];
  m_mtx_algo.unlock();

  algo->onSweepBegin(this);
}

void Collector::onSweepComplete( int64 storedMem, int64 storedCount, int64 freedMem, int64 freedItems )
{
   m_mtx_accountmem.lock();
   m_storedMem = storedMem;
   m_storedItems = storedCount;
   m_aliveMem -= freedMem;
   m_aliveItems -= freedItems;
   m_sweepLoops ++;
   m_mtx_accountmem.unlock();

   m_mtx_algo.lock();
   CollectorAlgorithm* algo = m_algo[m_curAlgoID];
   m_mtx_algo.unlock();

   algo->onSweepComplete( this, freedMem, freedItems );
}




//==========================================================================================
// Marker
//

void* Collector::Marker::run()
{
   MESSAGE( "Collector::Marker::run -- starting" );

   Event& evt = m_master->_p->m_markerWork;
   PoolFIFO& commands = m_master->_p->m_markCommands;

   bool isAlive = true;
   while( isAlive )
   {
      Cmd* command = commands.tdeq<Cmd>();
      if( command == 0 )
      {
         evt.wait(-1);
         continue;
      }

      bool clearCmd = true;
      switch( command->m_type )
      {
      case Cmd::e_cmd_register:
         // register will be always performed
         performRegister(command->m_ctx);
         break;

      case Cmd::e_cmd_unregister:
         // unregister will be always performed
         performUnregister(command->m_ctx);
         break;

      case Cmd::e_cmd_mark:
         clearCmd = performMark(command);
         break;

      case Cmd::e_cmd_offer:
         performOffer(command->m_ctx);
         break;

      case Cmd::e_cmd_check:
         clearCmd = performCheck(command);
         break;

      case Cmd::e_cmd_fullgc:
         clearCmd = performFull(command);
         break;

      case Cmd::e_start_mark:
         clearCmd = performStartMark(command);
         break;

      case Cmd::e_sweep_complete:
         // always succeed
         performSweepComplete();
         break;

      case Cmd::e_cmd_enable:
         performEnable();
         break;

      case Cmd::e_cmd_disable:
         performDisable();
         break;

      case Cmd::e_cmd_terminate:
         performTerminate();
         isAlive = false;
         break;

      case Cmd::e_cmd_abort:
         isAlive = false;
         break;

      default:
         TRACE( "Collector::Marker::run -- received an unknown command %d -- should not happen.", (int) command->m_type);
         break;
      }

      if ( clearCmd )
      {
         command->signal();
         command->vdispose();  // will decref context and resources as necessary.
      }
   }

   m_state = e_state_terminated;
   MESSAGE( "Collector::Marker::run -- stopping" );
   return 0;
}


void Collector::Marker::performRegister(VMContext* ctx)
{
   if( m_master->_p->m_contexts.insert(ctx).second )
   {
      TRACE("Collector::Marker::performRegister(%p) inserted context id = (%d:%d)", ctx, ctx->process()->id(), ctx->id() );
      ctx->incref();

      // are we currently performing a mark-inspect request?
      if( m_state == e_state_inspecting )
      {
         MESSAGE("Collector::Marker::performRegister -- currently waiting on inspection rendez-vous.");
         m_master->_p->m_inspectedContexts.insert(ctx);
         ctx->setInspectEvent();
      }
   }
   else {
      TRACE("Collector::Marker::performRegister(%p) discarded context id = (%d:%d)", ctx, ctx->process()->id(), ctx->id() );
   }
}


void Collector::Marker::performUnregister(VMContext* ctx)
{
   if( m_master->_p->m_contexts.erase(ctx) )
   {
      TRACE("Collector::Marker::performUnregister(%p) erased context id = (%d:%d)", ctx, ctx->process()->id(), ctx->id() );
      ctx->decref();

      if ( m_master->_p->m_inspectedContexts.erase(ctx) )
      {
         TRACE("Collector::Marker::performUnregister -- context %p (%d:%d) was scheduled for inspection", ctx, ctx->process()->id(), ctx->id() );
         if( m_master->_p->m_inspectedContexts.empty() )
         {
            MESSAGE("Collector::Marker::performUnregister -- inspectd contexts now emty, sending a start mark request" );

            // It is possible that another newly registered context is travelling in the command queue
            // we can start marking ONLY:
            //   - if the inspected set is empty AND
            //   - if the start_mark message emerges to the main loop.
            m_master->_p->sendMarkMessage(Cmd::e_start_mark);
         }
      }
   }
   else {
      TRACE("Collector::Marker::performRegister(%p) ignored context id = (%d:%d)", ctx, ctx->process()->id(), ctx->id() );
   }
}


void Collector::Marker::performOffer(VMContext* ctx)
{
   if( m_master->_p->m_inspectedContexts.erase(ctx) && m_state == e_state_inspecting )
   {
      TRACE("Collector::Marker::performOffer(%p) context (%d:%d) was waited.", ctx, ctx->process()->id(), ctx->id() );

      if ( m_master->_p->m_inspectedContexts.empty() )
      {
         // send a message for ourselves.
         MESSAGE("Collector::Marker::performOffer -- All waited context have been offered; requesting start of mark loop.");

         // It is possible that another newly registered context is travelling in the command queue
         // we can start marking ONLY:
         //   - if the inspected set is empty AND
         //   - if the start_mark message emerges to the main loop.

         m_master->_p->sendMarkMessage(Cmd::e_start_mark);
      }
   }
   else {
      TRACE("Collector::Marker::performOffer(%p) ignored context id = (%d:%d)", ctx, ctx->process()->id(), ctx->id() );
      ctx->resetInspectEvent();
      ctx->process()->vm()->contextManager().onContextDescheduled(ctx);
   }
}


bool Collector::Marker::performMark( Cmd* cmd )
{
   MESSAGE("Collector::Marker::performMark");
   if ( m_state != e_state_idle )
   {
      MESSAGE("Collector::Marker::performMark -- not in idle state, saving the incoming CMD for later.");
      m_master->_p->m_markDelayed.enqueue(cmd);
      return false; // don't dispose.
   }

   askMark();
   // we don't set any mark type, as it should have been reset to e_mark_justmark after last operation.

   Private::ContextSet& i_set = m_master->_p->m_inspectedContexts;
   if( ! i_set.empty() && cmd->isSignalable() )
   {
      m_master->_p->m_markWaiters.enqueue(cmd);
      return false;
   }
   // we can dispose.
   return true;
}

bool Collector::Marker::performFull( Cmd* cmd )
{
   MESSAGE("Collector::Marker::performFull");
   if ( m_state != e_state_idle )
   {
      MESSAGE("Collector::Marker::performFull -- not in idle state, saving the incoming CMD for later.");
      m_master->_p->m_markDelayed.enqueue(cmd);
      return false; // don't dispose.
   }

   askMark();

   Private::ContextSet& i_set = m_master->_p->m_inspectedContexts;
   if( ! i_set.empty() )
   {
      // there's work to do -- and we want a sweep to be performed.
      m_mark_mode = e_mark_full;

      // and eventually to be told to the waiter.
      if( cmd->isSignalable() )
      {
         m_master->_p->m_sweepWaiters.enqueue(cmd);
         return false;
      }
   }
   // we can dispose.
   return true;
}


bool Collector::Marker::performCheck( Cmd* cmd )
{
   MESSAGE("Collector::Marker::performCheck");

   if ( m_state != e_state_idle )
   {
      if( m_state == e_state_inspecting || m_state == e_state_marking )
      {
         MESSAGE("Collector::Marker::performCheck -- Already checking, ignoring a check request.");
         if( cmd->isSignalable())
         {
            // but signal when done if necessary.
            m_master->_p->m_markWaiters.enqueue(cmd);
         }
         else
         {
            return true;
         }
      }
      else{
         MESSAGE("Collector::Marker::performCheck -- not in idle state, saving the incoming CMD for later.");
         m_master->_p->m_markDelayed.enqueue(cmd);
      }
      return false; // don't dispose.
   }

   askMark();
   // promote just_mark, but don't demote mark_full
   Private::ContextSet& i_set = m_master->_p->m_inspectedContexts;
   if( ! i_set.empty() )
   {
      // there's work to do -- we will check if a sweep is in order.
      if( m_mark_mode == e_mark_justmark )
      {
         // promote just mark, but don't demote mark_full.
         // shouldn't be needed, as we're in idle state, and in idle state mark_mode is always just check
         m_mark_mode = e_mark_check;
      }

      if( cmd->isSignalable() )
      {
         // enqueue in mark waiters.
         m_master->_p->m_markWaiters.enqueue(cmd);
         return false;
      }
   }

   return true;
}


void Collector::Marker::rollover()
{
   MESSAGE("Collector::rollover -- start");

   // we work under the hypotesis that there can't be a mark loop running during a sweep loop.
   GCToken* token = m_master->m_garbageRoot->m_next;
   GCToken* end = m_master->m_garbageRoot;
   while( token != end )
   {
      Class* cls = token->m_cls;
      void* data = token->m_data;
      cls->gcMarkInstance(data, 0);
      token = token->m_next;
   }

   MESSAGE("Collector::rollover -- complete");
}


void Collector::Marker::performTerminate()
{
   MESSAGE("Collector::Marker::performTerminate");

   Private::ContextSet& set = m_master->_p->m_contexts;
   Private::ContextSet& i_set = m_master->_p->m_inspectedContexts;

   // clearing the set here results in preventing incoming offered contexts to trigger a mark loop.
   i_set.clear();

   // unregisters all the contexts
   Private::ContextSet::iterator iter = set.begin();
   Private::ContextSet::iterator end = set.end();
   while( iter != end )
   {
      VMContext* ctx = *iter;
      TRACE("Collector::Marker::performTerminate unregistering context %p (%d:%d)", ctx, ctx->process()->id(), ctx->id());
      ctx->decref();
      ++iter;
   }
   set.clear();

   // notify the waiters, we won't be around anymore.
   m_master->_p->clearQueue( m_master->_p->m_markCommands );
   m_master->_p->clearQueue( m_master->_p->m_markDelayed );
}


void Collector::Marker::askMark()
{
   Private::ContextSet& set = m_master->_p->m_contexts;
   Private::ContextSet& i_set = m_master->_p->m_inspectedContexts;

   // should be cleared already, but...
   if( i_set.empty() )
   {

      Private::ContextSet::const_iterator iter = set.begin();
      Private::ContextSet::const_iterator end = set.end();
      while( iter != end )
      {
         VMContext* ctx = *iter;
         ctx->setInspectEvent();
         i_set.insert(ctx);
         ++iter;
      }

      if( ! i_set.empty() )
      {
         m_state = e_state_inspecting;
      }

      TRACE1("Collector::Marker::askMark -- inspecting %u contexts", (unsigned int)i_set.size() );
   }
   else
   {
      MESSAGE("Collector::Marker::performMark -- already inspecting.");
   }
}


bool Collector::Marker::performStartMark( Cmd* cmd )
{
   // do we have a sweep in progress?
   if ( m_state != e_state_inspecting )
   {
      // then delay our mark
      MESSAGE("Collector::Marker::performStartMark -- delayed because sweep is in progress." );
      m_master->_p->m_markDelayed.enqueue(cmd);
      return false;
   }
   else if ( m_master->_p->m_inspectedContexts.empty() )
   {
      // The request to start marking has emerged in the main loop,
      // AND no other context is inbound for inspection, so we can start marking.
      m_state = e_state_marking;

      MESSAGE("Collector::Marker::performStartMark -- proceeding to mark loop now." );
      markLoop();
      releaseContexts();

      if( m_mark_mode == e_mark_check )
      {
         // promote or demote the mark mode depending on what we found.
         MESSAGE("Collector::Marker::performStartMark -- checking if the updated memory levels require a sweep." );

         m_master->m_mtx_algo.lock();
         CollectorAlgorithm* algo = m_master->m_algo[m_master->m_curAlgoID];
         m_master->m_mtx_algo.unlock();

         m_mark_mode = algo->onCheckComplete(m_master) ? e_mark_full : e_mark_justmark;
      }

      // shall we proceed to full gc?
      if( m_mark_mode == e_mark_full )
      {
         MESSAGE("Collector::Marker::performStartMark -- sweep is requested, starting sweep." );
         m_state = e_state_sweeping;
         // reset mark mode now
         m_mark_mode = e_mark_justmark;

         m_master->_p->sendSweepMessage(Cmd::e_cmd_fullgc);
      }
      else {
         MESSAGE("Collector::Marker::performStartMark -- sweep not requested, going idle." );
         goToIdle();
      }

      return true;
   }
   else {
      MESSAGE("Collector::Marker::performStartMark -- inspect not complete, reiterating the start request." );
      m_master->_p->m_markCommands.enqueue(cmd);
      return false;
   }
}


void Collector::Marker::performSweepComplete()
{
   MESSAGE("Collector::Marker::performSweepComplete -- acknowledged end of sweep." );
   goToIdle();
}


void Collector::Marker::performEnable()
{

   m_bPendingDisable = false;
   if( m_state == e_state_disable )
   {
      MESSAGE("Collector::Marker::performEnable -- enabling." );
      goToIdle();
   }
   else {
      MESSAGE("Collector::Marker::performEnable -- currently not disabled, clearing disable pending request." );
   }
}

void Collector::Marker::performDisable()
{
   if( m_state == e_state_inspecting )
   {
      MESSAGE("Collector::Marker::performDisable -- disabling during the inspecting status." );
      m_state = e_state_disable;
      releaseContexts();
   }
   else if( m_state == e_state_idle )
   {
      MESSAGE("Collector::Marker::performDisable -- disabling in idle." );
      m_state = e_state_disable;
   }
   else {
      MESSAGE("Collector::Marker::performDisable -- setting a disable request for later fulfilling." );
      m_bPendingDisable = true;
   }
}



void Collector::Marker::goToIdle()
{
   MESSAGE("Collector::Marker::goToIdle -- Entering idle state." );

   if( m_bPendingDisable )
   {
      m_state = e_state_disable;
      m_bPendingDisable = false;
   }

   m_state = e_state_idle;

   // Move the delayed command back to the command queue
   PoolFIFO& delayed = m_master->_p->m_markDelayed;
   PoolFIFO& commands = m_master->_p->m_markCommands;

   while( ! delayed.empty() )
   {
      Cmd* cmd = delayed.tdeq<Cmd>();
      commands.enqueue(cmd);
   }
   // we'll be checking the command queue before sensing the event, no need to set it.
}


void Collector::Marker::releaseContexts()
{
   Private::ContextSet& set = m_master->_p->m_contexts;
   Private::ContextSet::const_iterator iter = set.begin();
   Private::ContextSet::const_iterator end = set.end();

   while( iter != end )
   {
      VMContext* ctx = *iter;
      ctx->resetInspectEvent();
      ctx->process()->vm()->contextManager().onContextDescheduled(ctx);
      ++iter;
   }
}

void Collector::Marker::markLoop()
{
   // first, mark the locked items.
   m_master->m_mtx_currentMark.lock();
   uint32 mark = ++m_master->m_currentMark;
   if( mark >= MAX_GENERATION )
   {
      mark = m_master->m_currentMark = 1;
      m_master->m_mtx_currentMark.unlock();
      // before rollover, every alive object is above.
      // and we can't be sweeping during a mark loop.
      rollover();
   }
   else {
      m_master->m_mtx_currentMark.unlock();
   }

   m_master->markLocked(mark);

   GCToken* head = m_master->m_garbageRoot->m_next;

   // then, move the new ring.
   m_master->m_mtx_newRoot.lock();
   GCToken* first = m_master->m_newRoot->m_next;
   GCToken* last = m_master->m_newRoot->m_prev;
   m_master->m_newRoot->m_next = m_master->m_newRoot->m_prev = m_master->m_newRoot;
   m_master->m_mtx_newRoot.unlock();

   if( first != last )
   {
      head->m_next->m_prev = last;
      last->m_next = head->m_next;

      m_master->m_garbageRoot->m_next = head = first;
      first->m_prev = m_master->m_garbageRoot;
   }

   // Finally, ask for marking.
   Private::ContextSet& set = m_master->_p->m_contexts;
   Private::ContextSet::const_iterator iter = set.begin();
   Private::ContextSet::const_iterator end = set.end();
   while( iter != end )
   {
      VMContext* ctx = *iter;
      ctx->gcStartMark( mark );
      ctx->process()->modSpace()->gcMark(mark);
      ctx->process()->gcMark(mark);
      ctx->gcPerformMark();

      ++iter;
   }

   // mark complete; update accounting and notify the listeners.
   m_master->m_mtx_accountmem.lock();
   m_master->m_markLoops++;
   /*
   m_master->m_aliveItems = count;
   m_master->m_aliveMem = memory;
   */
   m_master->m_mtx_accountmem.unlock();

   // signal the waiters
   m_master->_p->clearQueue(m_master->_p->m_markWaiters);
}

//==========================================================================================
// Sweeper
//

void* Collector::Sweeper::run()
{
   MESSAGE( "Collector::Sweeper::run -- starting" );

   PoolFIFO &commands = m_master->_p->m_sweepCommands;
   Event& evt = m_master->_p->m_sweeperWork;

   bool isAlive = true;
   while( isAlive )
   {
      MESSAGE( "Collector::Sweeper::run -- waiting new activity" );

      Cmd* command = commands.tdeq<Cmd>();
      if( command == 0 )
      {
         evt.wait(-1);
         continue;
      }

      MESSAGE( "Collector::Sweeper::run -- new activity received, checking out" );

      bool clearCmd = true;
      switch( command->m_type )
      {
      case Cmd::e_cmd_fullgc:
         performFull();
         break;

      case Cmd::e_cmd_terminate:
         performTerminate();
         isAlive = false;
         break;

      case Cmd::e_cmd_abort:
         isAlive = false;
         break;

      default:
         TRACE( "Collector::Sweeper::run -- received an unprocessed command %d -- should not happen.",
                  (int) command->m_type);
         break;
      }

      if ( clearCmd )
      {
         command->signal();
         command->vdispose();  // will decref context and resources as necessary.
      }
   }

   MESSAGE( "Collector::Sweeper::run -- stopping" );
   return 0;
}

void Collector::Sweeper::performFull()
{
   TRACE( "Collector::Sweeper::performFull -- Collection for %d/%d", m_master->m_currentMark, m_lastSweepMark );
   // changes to oldest mark are made visible via waits on sweeper work.
   if( m_lastSweepMark == m_master->m_currentMark )
   {
      // we already swept
      MESSAGE1( "Ignoring request" );
   }
   else
   {
      sweep( m_master->m_currentMark );
      m_lastSweepMark = m_master->m_currentMark;
   }

   // inform the waiters that we have been working
   // it's ok to signal a waiter that came in the meanwhile
   MESSAGE1( "Collector::Sweeper::performFull -- signaling waiters" );
   PoolFIFO& waiters = m_master->_p->m_sweepWaiters;
   m_master->_p->clearQueue(waiters);

   // notifying the marker that we're done sweeping.
   MESSAGE1( "Collector::Sweeper::performFull -- Notifing the marker." );
   m_master->_p->sendMarkMessage(Cmd::e_sweep_complete);
}


void Collector::Sweeper::performTerminate()
{
   MESSAGE("Collector::Marker::performTerminate");
   m_master->_p->clearQueue(m_master->_p->m_sweepWaiters);
}


void Collector::Sweeper::sweep( uint32 lastGen )
{
   TRACE( "Collector::Sweeper::sweep -- sweeping prior to %d", lastGen );

   // disengage the ring.
   m_master->m_mtx_garbageRoot.lock();
   GCToken* ring = m_master->m_garbageRoot->m_next;

   if( ring == m_master->m_garbageRoot )
   {
      m_master->m_mtx_garbageRoot.unlock();
      MESSAGE( "Collector::Sweeper::sweep -- Nothing to do");
      return;
   }

   GCToken* begin = ring;
   // the root never changes so we can reference it outside the lock.
   GCToken* end = m_master->m_garbageRoot->m_prev;
   ring->m_prev = 0;
   end->m_next = 0; // ok even if the we have a single root
   m_master->m_garbageRoot->m_next = m_master->m_garbageRoot;
   m_master->m_garbageRoot->m_prev = m_master->m_garbageRoot;

   m_master->m_mtx_garbageRoot.unlock();

   m_master->onSweepBegin();

   int64 freedMem = 0;
   int64 freedCount = 0;

   int64 storedMem = 0;
   int64 storedCount = 0;

   int32 priority = 0;
   int32 maxPriority = 0;

   while( priority <= maxPriority )
   {
      while( ring != 0 )
      {
         Class* cls = ring->cls();
         void* data = ring->data();

         if ( ! cls->gcCheckInstance(data, lastGen ) )
         {
            if( cls->clearPriority() > priority )
            {
               // skip this, we need to check this later.
               if( cls->clearPriority() > maxPriority ) {
                  maxPriority = cls->clearPriority();
               }
               ring = ring->m_next;
            }
            else
            {
               // time to collect it now.
      #ifndef NDEBUG
               String temp;
               try {
                  Item temp(cls, data);
                  TRACE2( "Collector::Sweeper::sweep -- killing %s (%p) of class %s",
                        temp.describe(0,60).c_ize(), data, cls->name().c_ize() );
               }
               catch( Error* e )
               {
                  TRACE2( "Collector::Sweeper::sweep -- while describing instance %p of class %s: %s",
                        data, cls->name().c_ize(), e->describe(true).c_ize() );
                  e->decref();
               }
      #endif

               #if FALCON_TRACE_GC
               if( m_master->m_bTrace ) {
                  m_master->onDestroy( data );
               }
               #endif

               freedMem += cls->occupiedMemory( data );
               freedCount++;
               cls->dispose( data );

               // unlink the ring
               if( ring == begin ) {
                  begin = begin->m_next;
                  // no need to reset begin->m_prev
               }
               else {
                  ring->m_prev->m_next = ring->m_next;
               }

              if( ring == end ) {
                  end = end->m_prev;
                  // need to reset end->m_next in case of another priority loop.
                  if( end != 0 )
                  {
                     end->m_next = 0;
                  }
               }
               else {
                  ring->m_next->m_prev = ring->m_prev;
               }

              GCToken* current = ring;
              ring = ring->m_next;

               m_master->disposeToken(current);
            }
         }
         // else, it's not time for collection.
         else {
            // should contabilize now?
            if( cls->clearPriority() == priority )
            {
               storedCount++;
               storedMem += cls->occupiedMemory( data );
            }

            ring = ring->m_next;
         }
      }

      // reset the ring.
      ring = begin;
      ++priority;
   }

   // put the ring back in place.
   if( begin != 0 ) {
      m_master->m_mtx_garbageRoot.lock();
      GCToken* next = m_master->m_garbageRoot->m_next;
      begin->m_prev = m_master->m_garbageRoot;
      end->m_next = next;
      next->m_prev = end;
      m_master->m_garbageRoot->m_next = begin;
      m_master->m_mtx_garbageRoot.unlock();
   }

   TRACE( "Collector::Sweeper::sweep -- reclaimed %d items, %d bytes",
            (int) freedCount, (int) freedMem );

   m_master->onSweepComplete( storedMem, storedCount, freedMem, freedCount );
}


//==========================================================================================
// Timer
//

void* Collector::Timer::run()
{
   MESSAGE( "Collector::Timer::run -- starting" );

   Mutex& mtxTimer = m_master->m_mtx_timer;
   Event& work = m_master->m_timerWork;
   int64& randezVous = m_master->m_algoRandezVous;

   while ( atomicFetch(m_master->m_aLive) )
   {
      MESSAGE( "Collector::Sweeper::run -- waiting new activity" );
      int64 nextRandezVous;
      mtxTimer.lock();
      nextRandezVous = randezVous;
      mtxTimer.unlock();

      int64 now = Sys::_milliseconds();
      if( nextRandezVous > 0 && now > nextRandezVous )
      {
         m_master->currentAlgorithmObject()->onTimeout( m_master );
         nextRandezVous = -1;
      }

      int64 waitTime = nextRandezVous > 0 ? nextRandezVous - now : -1;

      work.wait((int32)waitTime);
   }

   return 0;
}


//===============================================================================================
// Support for tracing GC allocations
// (Must be enabled at compile time, but it will work also in release mode).
//


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
      if (! cls->hasSharedInstances() )
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
      else {
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
      if( ! r( status ) )
      {
         break;
      }

      ++iter;
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
