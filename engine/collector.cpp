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

/*
 * Tokens posted by entities that explicitly request for full GC markings.
 *
 * Full GC Marks are done by saving all the VMContests that need to be marked,
 * and an event that is set when all the contexts are processed.
 *
 * \note: when a context is de-registered, all the pending mark tokens must be
 * scanned to remove the context from them.
 */
class MarkToken
{
public:
   typedef std::set<VMContext*> ContextSet;
   ContextSet m_set;

   Event* m_toBeSignaled;

   MarkToken( Event* evt ):
      m_toBeSignaled(evt)
   {}

   ~MarkToken()
   {
      ContextSet::iterator iter = m_set.begin();
      while( iter != m_set.end() )
      {
         VMContext* ctx = *iter;
         ctx->decref();
         ++iter;
      }

      signal();
   }


   void signal()
   {
      if (m_toBeSignaled != 0 )
      {
         m_toBeSignaled->set();
         m_toBeSignaled = 0;
      }
   }


   void onContextProcessed( VMContext* ctx )
   {
      ContextSet::iterator iter = m_set.find(ctx);

      if ( iter != m_set.end() )
      {
         m_set.erase(iter);
         ctx->decref();
      }

      if ( m_set.empty() )
      {
         signal();
      }
   }
};


class Collector::Private
{
public:
   // This mutex is used for all the context modify operations,
   // - m_contexts
   // - m_markTokens
   // - Context related counters and variables in the main Collector class.
   //
   Mutex m_mtx_contexts;

   typedef std::map<uint32, VMContext*> ContextMap;
   ContextMap m_contexts;

   Mutex m_mtx_markingList;
   typedef std::deque<VMContext*> MarkingList;
   MarkingList m_markingList;

   // This list is modified inside the parent m_mtxRequest
   typedef std::deque<Event*> EventList;
   EventList m_sweepTokens;

   typedef std::deque<MarkToken*> MarkTokenList;
   MarkTokenList m_markTokens;

#if FALCON_TRACE_GC
   typedef std::map<void*, Collector::DataStatus* > HistoryMap;
   HistoryMap m_hmap;
#endif

   Private() {}
   ~Private()
   {
      clearTrace();

      EventList::iterator eli = m_sweepTokens.begin();
      while( eli != m_sweepTokens.end() )
      {
         (*eli)->set();
         ++eli;
      }
      m_sweepTokens.clear();

      MarkTokenList::iterator mti = m_markTokens.begin();
      while( mti != m_markTokens.end() )
      {
         MarkToken* mt = *mti;
         delete mt;
         ++mti;
      }
      m_markTokens.clear();
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
   m_thMonitor(0),
   m_monitor(this),
   m_thMarker(0),
   m_marker(this),
   m_thSweeper(0),
   m_sweeper(this),
   m_sweepPerformed(false),

   m_aLive(1),
   m_bTrace( false ),
   m_currentMark(0),
   m_oldestMark(0),
   m_storedMem(0),
   m_storedItems(0),
   _p( new Private )
{
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
   m_ramp  = new CollectorAlgorithm*[ FALCON_COLLECTOR_ALGORITHM_COUNT ];
   m_ramp[FALCON_COLLECTOR_ALGORITHM_OFF] = new CollectorAlgorithmNone;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_FIXED] = new CollectorAlgorithmFixed(1000000,5000000,1000000);
   m_ramp[FALCON_COLLECTOR_ALGORITHM_STRICT] = new CollectorAlgorithmStrict;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_SMOOTH] = new CollectorAlgorithmSmooth;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_LOOSE] = new CollectorAlgorithmLoose;

   // force initialization in rampMode by setting a different initial value;
   /*
   m_curRampID = FALCON_COLLECTOR_ALGORITHM_DEFAULT;
   m_curRampMode = m_ramp[m_curRampID];
   */
   m_curRampID = FALCON_COLLECTOR_ALGORITHM_FIXED;
   m_curRampMode = m_ramp[m_curRampID];
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
      delete m_ramp[ri];
   }
   delete _p;
}


bool Collector::rampMode( int mode )
{
   if( mode >= 0 && mode < FALCON_COLLECTOR_ALGORITHM_COUNT )
   {
      m_mtx_ramp.lock();
      if ( m_curRampID != mode )
      {
         m_curRampID = mode;
         m_curRampMode = m_ramp[mode];
         //TODO: reset
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


void Collector::registerContext( VMContext *ctx )
{
   TRACE( "Collector::registerContext - %p(%d) in Process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );
   ctx->incref();

   _p->m_mtx_contexts.lock();
   uint32 mark = ++m_currentMark;
   if( _p->m_contexts.empty() ) {
      m_oldestMark = mark;
   }
   ctx->gcStartMark(mark);
   _p->m_contexts[ctx->currentMark()]= ctx;
   _p->m_mtx_contexts.unlock();

   // notice that when registered the context should not have any
   // item to be marked -- as it's registered at creation.
}


void Collector::unregisterContext( VMContext *ctx )
{
   TRACE( "Collector::unregisterContext - %p(%d) in Process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );

   _p->m_mtx_contexts.lock();
   bool erased = _p->m_contexts.erase(ctx->currentMark()) != 0;

   // also, be sure that we're not waiting for this context to be marked.
   Private::MarkTokenList::iterator mti = _p->m_markTokens.begin();
   Private::MarkTokenList::iterator mtend = _p->m_markTokens.end();
   // notice that token lists are very rarely used.
   while( mti != mtend )
   {
      MarkToken* token = *mti;
      // we don't need to unlock even if ctx gets decreffed,
      // as we hold a reference and we know the context won't be destroyed
      token->onContextProcessed(ctx);
      ++mti;
   }
   _p->m_mtx_contexts.unlock();

   if( erased ) {
      ctx->decref();
   }
}


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
            disposeToken(prev);
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

   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );

   int64 memory = cls->occupiedMemory(data);
   m_mtx_accountmem.lock();
   m_storedItems++;
   m_storedMem+= memory;
   m_mtx_accountmem.unlock();

   // put the element in the new list.
   m_mtx_newRoot.lock();
   token->m_next =  m_newRoot ;
   token->m_prev =  m_newRoot->m_prev ;
   m_newRoot->m_prev->m_next =  token;
   m_newRoot->m_prev =  token;
   m_mtx_newRoot.unlock();

   return token;
}


GCLock* Collector::storeLocked( const Class* cls, void *data )
{
   // do we have spare elements we could take?
   GCToken* token = getToken( const_cast<Class*>(cls), data );
   
   GCLock* l = this->lock( token );

   int64 memory = cls->occupiedMemory(data);
   m_mtx_accountmem.lock();
   m_storedItems++;
   m_storedMem+= memory;
   m_mtx_accountmem.unlock();

   // put the element in the new list.
   m_mtx_newRoot.lock();
   token->m_next =  m_newRoot ;
   token->m_prev =  m_newRoot->m_prev ;
   m_newRoot->m_prev->m_next =  token;
   m_newRoot->m_prev =  token;
   m_mtx_newRoot.unlock();

   return l;
}


void Collector::performGC( bool wait )
{
   Event markEvt;
   Event evt;
   MarkToken markToken(&markEvt);

   // push a request for all the contexts to be marked.
   int32 count = 0;
   _p->m_mtx_contexts.lock();
   Private::ContextMap::iterator cti = _p->m_contexts.begin();
   Private::ContextMap::iterator ctend = _p->m_contexts.end();
   while( cti != ctend )
   {
      VMContext* ctx = cti->second;

      // ask the context to be inspected asap.
      ctx->setInspectEvent();
      // ^^ this might send the context to the monitor, and
      // that requires locking a mutex, check that is NEVER
      // locked against m_mtx_contexts.

      // if we don't wait, we won't post the markToken
      if( wait )
      {
         ctx->incref();
         markToken.m_set.insert(ctx);
      }
      ++cti;
      count++;

      // signal the marker.
      _p->m_markTokens.push_back(&markToken);
   }
   _p->m_mtx_contexts.unlock();

   if( wait )
   {
      // now we have to wait for the marking to be complete (?)
      if( count != 0 )
      {
         markEvt.wait(-1);
      }

      // Ask the sweeper to notify us when it's done.
      // The sweeper MIGHT have already done our sweep,
      // but it will notify us never the less even if it doesn't sweep again.
      m_mtxRequest.lock();
      _p->m_sweepTokens.push_back(&evt);
      m_mtxRequest.unlock();

      // ask the sweeper to work a bit,
      // so it will notify us even if it's done and currently idle.
      m_sweeperWork.set();
      evt.wait( -1 );
   }

   // we don't need to ask the sweeper to work if we don't wait,
   // the marker will do if necessary and when necessary.
}

//===================================================================================
// MT functions
//


void Collector::start()
{
   if ( m_thMonitor == 0 )
   {
      atomicSet(m_aLive, 1);
      m_thMonitor = new SysThread( &m_monitor );
      m_thMonitor->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
      m_thMarker = new SysThread( &m_marker );
      m_thMarker->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
      m_thSweeper = new SysThread( &m_sweeper );
      m_thSweeper->start( ThreadParams().stackSize( GC_THREAD_STACK_SIZE ) );
   }
}


void Collector::stop()
{
   if ( m_thMonitor != 0 )
   {
      atomicSet(m_aLive,0);

      // wake up our threads
      m_monitorWork.set();
      m_markerWork.set();
      m_sweeperWork.set();

      // join them
      void *dummy;
      m_thMonitor->join( dummy );
      m_thMarker->join( dummy );
      m_thSweeper->join( dummy );

      // and clear them
      m_thMonitor = 0;
      m_thMarker = 0;
      m_thSweeper = 0;
   }
}


bool Collector::offerContext( VMContext* ctx )
{
   TRACE( "Collector::offerContext -- being offeretd ctx %d(%p) in process %d(%p)",
            ctx->id(), ctx, ctx->process()->id(), ctx->process() );

   bool operate = false;

   if( ctx->markedForInspection() )
   {
      // when marked for inspection, we HAVE to accept it.
      operate = true;
   }
   else {
      m_mtx_ramp.lock();
      CollectorAlgorithm* algo = m_ramp[m_curRampID];
      m_mtx_ramp.unlock();

      t_status st = algo->checkStatus();

      switch( st ) {
      case e_status_green:
         MESSAGE1( "Collector::offerContext -- refusing because in green state." );
         break;

      case e_status_yellow:
         _p->m_mtx_contexts.lock();
         operate = m_oldestMark == ctx->currentMark();
         _p->m_mtx_contexts.unlock();

         TRACE1( "Collector::offerContext -- %s in yellow status.",
                  (operate?"accepting":"refusing"));
         break;

      case e_status_brown:
      case e_status_red:
         _p->m_mtx_contexts.lock();
         operate = (m_oldestMark == ctx->currentMark()) || ctx->markedForInspection();
         _p->m_mtx_contexts.unlock();

         TRACE1( "Collector::offerContext -- %s in %s status.",
                  (operate?"accepting":"refusing"), (st == e_status_red? "red":"brown" ));
         break;

      case e_status_required:
         operate = true;
         MESSAGE1("Collector::offerContext -- accepting because in required state.");
         break;
      }
   }

   if(operate) {
      ctx->incref();
      _p->m_mtx_markingList.lock();
      _p->m_markingList.push_back(ctx);
      _p->m_mtx_markingList.unlock();
      m_markerWork.set();
   }

   return operate;
}


void Collector::markNew( uint32 mark )
{
   // This method is called from the mark thread.
   MESSAGE( "Collector::markNew begin" );

   /*
   TextWriter ts(new StdOutStream);
   dumpHistory(&ts);
   */

   m_mtx_newRoot.lock();

   GCToken* newRingFront = m_newRoot->m_next;
   GCToken* newRingBack = 0;
   // now empty the ring.

   // the newRoot pointer never changes, so we can have it during an unlock.
   if ( newRingFront != m_newRoot )
   {
      newRingBack = m_newRoot->m_prev;
      // make the loop to turn into a list;
      newRingBack->m_next = 0;
      // disengage all the loop
      m_newRoot->m_next = m_newRoot;
      m_newRoot->m_prev = m_newRoot;
   }
   // else, there's nothing to do.
   m_mtx_newRoot.unlock();

   if ( newRingBack != 0 )
   {
      // this is the only thread that can change the current mark.
      TRACE( "Collector::markNew marking generation %u", mark );
      
      // first mark
      GCToken* newRing = newRingFront;
      int32 count = 0;
      while( newRing != 0 )
      {
         ++count;
#if FALCON_TRACE_GC
         if( m_bTrace ) {
            onMark(newRing->m_data);
         }
#endif
         newRing->m_cls->gcMarkInstance( newRing->m_data, mark );
         newRing = newRing->m_next;
      }

      // we can now insert our items in the garbageable items.
      m_mtx_garbageRoot.lock();
      newRingFront->m_prev = m_garbageRoot;
      newRingBack->m_next = m_garbageRoot->m_next;
      m_garbageRoot->m_next->m_prev = newRingBack;
      m_garbageRoot->m_next = newRingFront;
      m_mtx_garbageRoot.unlock();

      TRACE1( "Collector::markNew Marked generation %d (%d items)", mark, count );
   }
   else {
      MESSAGE( "Collector::markNew nothing to do" );
   }
}



// WARNING: Rollover is to be called by the Marker agent
void Collector::rollover()
{
   MESSAGE("Collector::rollover -- start");

   // we have to re-assign and re-mark all the contexts.
   // we're not changing the marks of the item; we'll let next mark loops to do that.
   uint32 baseMark = 1;

   std::deque<VMContext* > liveCtx;

   _p->m_mtx_contexts.lock();
   Private::ContextMap::iterator iter = _p->m_contexts.begin();
   Private::ContextMap::iterator end = _p->m_contexts.end();

   while( iter != end ) {
      VMContext* ctx = iter->second;
      liveCtx.push_back(ctx);
      ctx->gcStartMark(baseMark++);
      ++iter;
   }

   _p->m_contexts.clear();

   std::deque<VMContext* >::iterator li = liveCtx.begin();
   std::deque<VMContext* >::iterator lie = liveCtx.end();

   while( li != lie ) {
      VMContext* ctx = *li;
      _p->m_contexts[ctx->currentMark()] = ctx;
      ++li;
   }
   // at worst, we can just have a sweep loop running without doing nothing.
   m_oldestMark = 1;
   m_currentMark = baseMark-1;  // ok even if 0, currentMark is pre-advanced
   _p->m_mtx_contexts.unlock();

   TRACE("Collector::rollover -- marked %d contexts -- notifying sweeper", baseMark-1);

   // prior letting the marker to proceed, we must be sure that the sweeper
   // aknowledges the change.
   Event evt;
   m_mtxRequest.lock();
   _p->m_sweepTokens.push_back(&evt);
   m_mtxRequest.unlock();

   m_sweeperWork.set();
   evt.wait( -1 );

   MESSAGE("Collector::rollover -- sweeper notified");
}


// WARNING: Rollover is to be called by the Marker agent
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
         const Item& item = lock->item();
         Class* cls;
         void* data;
         if( item.asClassInst(cls, data ) ){
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


void Collector::accountMemory( int64 memory )
{
   m_mtx_accountmem.lock();
   m_storedMem += memory;
   m_mtx_accountmem.unlock();
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

void Collector::stored( int64& memory, int64& items ) const
{
   m_mtx_accountmem.lock();
   memory = m_storedMem;
   items = m_storedItems;
   m_mtx_accountmem.unlock();
}

void Collector::onSweepBegin()
{
  m_mtx_ramp.lock();
  CollectorAlgorithm* algo = m_ramp[m_curRampID];
  m_mtx_ramp.unlock();

  algo->onSweepBegin();
}

void Collector::onSweepComplete( int64 freedMem, int64 freedItems )
{
   m_mtx_accountmem.lock();
   m_storedMem -= freedMem;
   m_storedItems -= freedItems;
   int64 mem = m_storedMem;
   int64 items = m_storedItems;
   m_mtx_accountmem.unlock();

   m_mtxRequest.lock();
   m_sweepPerformed = true;
   m_mtxRequest.unlock();

   m_mtx_ramp.lock();
   CollectorAlgorithm* algo = m_ramp[m_curRampID];
   m_mtx_ramp.unlock();

   algo->onSweepComplete( mem, items );
}


//==========================================================================================
// Monitor
//

void* Collector::Monitor::run()
{
   MESSAGE( "Collector::Monitor::run -- starting" );

   Collector::Private::ContextMap& contexts = m_master->_p->m_contexts;
   Mutex& mtxContexts = m_master->_p->m_mtx_contexts;
   Event& work = m_master->m_monitorWork;

   // start from green status.
   Collector::t_status prev_st = Collector::e_status_green;

   while( atomicFetch(m_master->m_aLive) )
   {

      work.wait(GC_IDLE_TIME);

      m_master->m_mtx_ramp.lock();
      Collector::t_status st = m_master->m_ramp[m_master->m_curRampID]->checkStatus();
      m_master->m_mtx_ramp.unlock();

      /*
      TRACE( "Collector::Monitor::run -- Working %ld on %ld items (in mode %d)",
               (long)m_master->storedMemory(), (long) m_master->storedItems(), st );
      */
      if( st == Collector::e_status_red || st == Collector::e_status_required )
      {
         bool perform = false;
         if( prev_st != st ) {
            TRACE1( "Collector::Monitor::run -- sending block requests for status change %d -> %d",
                           prev_st, st );

            perform = true;
         }
         else {
            m_master->m_mtxRequest.lock();
            bool swept = m_master->m_sweepPerformed;
            m_master->m_mtxRequest.unlock();

            if( swept ) {
               MESSAGE( "Collector::Monitor::run -- sending block requests after (failed) sweep" );
               perform = true;
            }
         }

         if( perform )
         {
            // send all the blocking inspect requests to all the contexts.
            mtxContexts.lock();
            Collector::Private::ContextMap::iterator iter = contexts.begin();
            Collector::Private::ContextMap::iterator end = contexts.end();
            while( iter != end ) {
               //TODO!!!
               //VMContext* ctx = iter->second;
               //ctx->setInspectEvent();
               ++iter;
            }
            mtxContexts.unlock();

            m_master->m_mtxRequest.lock();
            m_master->m_sweepPerformed = false;
            m_master->m_mtxRequest.unlock();
         }
      }

      prev_st = st;
   }

   MESSAGE( "Collector::Monitor::run -- stopping" );
   return 0;
}


//==========================================================================================
// Marker
//

void* Collector::Marker::run()
{
   MESSAGE( "Collector::Marker::run -- starting" );

   atomic_int& aLive = m_master->m_aLive;
   Event& work = m_master->m_markerWork;

   Collector::Private::MarkingList& markingList = m_master->_p->m_markingList;
   Mutex& mtxList = m_master->_p->m_mtx_markingList;

   Collector::Private::ContextMap& contexts = m_master->_p->m_contexts;
   Mutex& mtxContexts = m_master->_p->m_mtx_contexts;

   while( atomicFetch(aLive) )
   {
      work.wait(-1);
      mtxList.lock();
      while( ! markingList.empty() )
      {
         VMContext* toMark = markingList.front();
         markingList.pop_front();
         mtxList.unlock();

         TRACE( "Collector::Marker::run -- marking %d(%p) in process %d(%p)",
                  toMark->id(), toMark, toMark->process()->id(), toMark->process() );

         // we're the only one authorized to change the mark of a context
         uint32 oldMark = toMark->currentMark();
         uint32 newOldest = 0;

         mtxContexts.lock();
         // declare the new mark id inside the lock
         uint32 mark = ++ m_master->m_currentMark;
         contexts.erase( toMark->currentMark() );
         toMark->gcStartMark( mark );
         contexts[mark] = toMark;
         if( oldMark == m_master->m_oldestMark ) {
            newOldest = contexts.begin()->first;
         }
         mtxContexts.unlock();

         // continue marking
         toMark->gcPerformMark();
         // declare the mark is marked
         onMarked( toMark );

         // send the context back to the manager.
         toMark->clearEvents();  //TODO --- really?
         toMark->setInspectible(false);
         toMark->resetInspectEvent();
         toMark->vm()->contextManager().onContextDescheduled(toMark);

         TRACE1( "Collector::Marker::run -- mark complete %d(%p) in process %d(%p)",
                                    toMark->id(), toMark, toMark->process()->id(), toMark->process() );
         // we don't need the inspected context anymore.
         toMark->decref();

         // did we change the old mark?
         if( newOldest != 0 )
         {
            TRACE1( "Collector::Marker::run -- Abandoning oldest mark %d (now %d)",
                     oldMark, newOldest );
            // time to mark the gclocked items...
            m_master->markLocked( mark );

            //... and the new items ...
            m_master->markNew( mark );

            // and finally update the new oldest.
            // Notice that this is the only thread that can change the oldest mark.
            mtxContexts.lock();
            m_master->m_oldestMark = newOldest;
            mtxContexts.unlock();

            // signal the sweeper to have a look.
            m_master->m_sweeperWork.set();
         }

         // lastly, check if we need to rollover the mark counter.
         if( mark > MAX_GENERATION )
         {
            m_master->rollover();
         }

         mtxList.lock();
      }
      mtxList.unlock();
   }

   MESSAGE( "Collector::Marker::run -- stopping" );
   return 0;
}


void Collector::Marker::onMarked( VMContext* ctx )
{
   m_master->_p->m_mtx_contexts.lock();
   // also, be sure that we're not waiting for this context to be marked.
   Private::MarkTokenList::iterator mti = m_master->_p->m_markTokens.begin();
   Private::MarkTokenList::iterator mtend = m_master->_p->m_markTokens.end();
   // notice that token lists are very rarely used.
   while( mti != mtend )
   {
      MarkToken* token = *mti;
      // we don't need to unlock even if ctx gets decreffed,
      // as we hold a reference and we know the context won't be destroyed
      token->onContextProcessed(ctx);
      ++mti;
   }
   m_master->_p->m_mtx_contexts.unlock();
}

//==========================================================================================
// Marker
//

void* Collector::Sweeper::run()
{
   MESSAGE( "Collector::Sweeper::run -- starting" );

   uint32 lastSweepMark = 0;
   Mutex& mtxContexts = m_master->_p->m_mtx_contexts;
   Collector::Private::EventList& sweepTokens = m_master->_p->m_sweepTokens;

   std::deque<Event*> toBeSignaled;

   while( atomicFetch(m_master->m_aLive) )
   {
      m_master->m_sweeperWork.wait(-1);
      bool bPerform = false;

      mtxContexts.lock();
      if( m_master->m_oldestMark != lastSweepMark ) {
         lastSweepMark = m_master->m_oldestMark;
         if( lastSweepMark > 1 )
         {
            bPerform = true;
         }

         // we want to signal even if we don't perform.
         if( ! sweepTokens.empty() )
         {
            toBeSignaled = sweepTokens;
            sweepTokens.clear();
         }
      }
      mtxContexts.unlock();

      // should we perform sweep?
      if( bPerform )
      {
         sweep( lastSweepMark );
      }

      // signal the waiters that we're done.
      std::deque<Event*>::iterator signal_iter = toBeSignaled.begin();
      while( signal_iter != toBeSignaled.end() ) {
         Event* evt = *signal_iter;
         evt->set();
         ++signal_iter;
      }
      toBeSignaled.clear();
   }

   MESSAGE( "Collector::Sweeper::run -- stopping" );
   return 0;
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
            }
            else
            {
               // time to collect it now.
      #ifndef NDEBUG
               String temp;
               cls->describe(data, temp);
               TRACE2( "Collector::Sweeper::sweep -- killing %s (%p) of class %s",
                        temp.c_ize(), data, cls->name().c_ize() );
      #endif

               #if FALCON_TRACE_GC
               if( m_master->m_bTrace ) {
                  m_master->onDestroy( data );
               }
               #endif

               freedMem += ring->m_cls->occupiedMemory( data );
               freedCount++;
               cls->dispose( data );

               // unlink the ring
               GCToken* current = ring;
               ring = ring->m_next;

               if( current == begin ) {
                  begin = begin->m_next;
                  // no need to reset begin->m_prev
               }
               else {
                  current->m_prev->m_next = current->m_next;
               }

               if( current == end ) {
                  end = end->m_prev;
                  // need to reset end->m_next in case of another priority loop.
                  end->m_next = 0;
               }
               else {
                  current->m_next->m_prev = current->m_prev;
               }

               m_master->disposeToken(current);
            }
         }
         // else, it's not time for collection.
         else {
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
      begin->m_prev = m_master->m_garbageRoot;
      end->m_next = m_master->m_garbageRoot->m_next;
      m_master->m_garbageRoot->m_next->m_prev = end;
      m_master->m_garbageRoot->m_next = begin;
      m_master->m_mtx_garbageRoot.unlock();
   }

   TRACE( "Collector::Sweeper::sweep -- reclaimed %d items, %d bytes",
            (int) freedCount, (int) freedMem );

   m_master->onSweepComplete( freedMem, freedCount );
}

}

/* end of collector.cpp */
