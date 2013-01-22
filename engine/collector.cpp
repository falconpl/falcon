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

#include <string>
#include <typeinfo>
#include <set>

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
   Mutex m_mtx_contexts;
   typedef std::set<VMContext*> ContextSet;
   ContextSet m_contexts;

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
   m_th(0),
   m_bLive(false),
   m_bRequestSweep( false ),
   m_bTrace( false ),
   m_currentMark(0),
   m_oldestMark(0),
   m_storedMem(0),
   m_storedItems(0),
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
   m_ramp  = new CollectorAlgorithm*[ FALCON_COLLECTOR_ALGORITHM_COUNT ];
   m_ramp[FALCON_COLLECTOR_ALGORITHM_OFF] = new CollectorAlgorithmNone;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_FIXED] = new CollectorAlgorithmFixed(10000000,50000000,100000000);
   m_ramp[FALCON_COLLECTOR_ALGORITHM_STRICT] = new CollectorAlgorithmStrict;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_SMOOTH] = new CollectorAlgorithmSmooth;
   m_ramp[FALCON_COLLECTOR_ALGORITHM_LOOSE] = new CollectorAlgorithmLoose;

   // force initialization in rampMode by setting a different initial value;
   m_curRampID = FALCON_COLLECTOR_ALGORITHM_DEFAULT;
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
   _p->m_contexts.insert(ctx);
   _p->m_mtx_contexts.unlock();

   ctx->gcMark( m_currentMark );
}


void Collector::unregisterContext( VMContext *ctx )
{
   TRACE( "Collector::unregisterContext - %p(%d) in Process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );

   _p->m_mtx_contexts.lock();
   bool erased = _p->m_contexts.erase(ctx) != 0;
   _p->m_mtx_contexts.unlock();

   if( erased ) {
      ctx->decref();
   }
}


void Collector::clearRing( GCToken *ringRoot )
{
   //TRACE( "Entering sweep %ld, allocated %ld", (long)gcmallocated(), (long)m_allocatedItems );
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

   //TRACE( "Sweeping step 1 complete %ld", (long)gcmallocated() );

   m_mtx_accountmem.lock();
   fassert( killed <= m_storedItems );
   m_storedItems -= killed;
   m_mtx_accountmem.unlock();

   TRACE( "Sweeping done, allocated %ld (killed %ld)", (long)m_storedItems, (long)killed );
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

   int64 memory = cls->occupiedMemory(data);
   m_mtx_accountmem.lock();
   m_storedItems++;
   m_storedMem+= memory;
   m_mtx_accountmem.unlock();

   // put the element in the new list.
   m_mtx_newitem.lock();
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

   int64 memory = cls->occupiedMemory(data);
   m_mtx_accountmem.lock();
   m_storedItems++;
   m_storedMem+= memory;
   m_mtx_accountmem.unlock();

   // put the element in the new list.
   m_mtx_newitem.lock();
   token->m_next =  m_newRoot ;
   token->m_prev =  m_newRoot->m_prev ;
   m_newRoot->m_prev->m_next =  token;
   m_newRoot->m_prev =  token;
   m_mtx_newitem.unlock();

   return l;
}


void Collector::gcSweep()
{

   //TRACE( "Sweeping %ld (mingen: %d, gen: %d)", (long)gcmallocated(), m_mingen, m_generation );

   m_mtx_ramp.lock();
   // ramp mode may change while we do the lock...
   //CollectorAlgorithm* rm = m_curRampMode;

   m_mtx_ramp.unlock();

   clearRing( m_garbageRoot );

   m_mtx_ramp.lock();
   //TODO: Call the ramp

   m_mtx_ramp.unlock();
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
      uint32 mark = m_currentMark;
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
   bool bMoreWork;

   while( m_bLive )
   {
      bMoreWork = false; // in case of sweep request, loop without pause.

      // first, detect the operating status. -- we don't care about races here.
      size_t memory = m_storedMem;
      int state = m_bRequestSweep || memory >= m_thresholdActive ? 2 :      // active mode
                  memory >= m_thresholdNormal ? 1 :      // normal mode
                  0;                                     // dormient mode

      TRACE( "Collector::run -- Working %ld on %ld items (in mode %d)",
               (long)memory, (long) m_storedItems, state );
      
      // if we have nothing to do, we shall wait a bit.
      if( ! bMoreWork )
      {
         MESSAGE( "Waiting GC idle time" );
         m_eRequest.wait(GC_IDLE_TIME);
      }
   }

   //TRACE( "Stopping %ld", (long)gcmallocated() );
   return 0;
}


// to be called with m_mtx_vms locked
void Collector::advanceGeneration( VMachine*, uint32 oldGeneration )
{
   uint32 curgen = ++m_currentMark;

   // detect here rollover.
   if ( curgen < oldGeneration || curgen >= MAX_GENERATION )
   {
      //TODO
      curgen = m_currentMark;
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
   uint32 mark = m_currentMark;
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
         item.gcMark( mark );
         
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


}

/* end of collector.cpp */


