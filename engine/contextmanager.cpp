/*
   FALCON - The Falcon Programming Language.
   FILE: contextmanager.cpp

   VM ContextManager managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/contextmanager.cpp"

#include <falcon/contextmanager.h>
#include <falcon/mt.h>
#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/sys.h>
#include <falcon/syncqueue.h>
#include <falcon/contextgroup.h>
#include <falcon/syncqueue.h>
#include "shared_private.h"

#include <map>
#include <set>
#include <vector>


namespace Falcon {

/**
 Message exchanged between the public methods and the manager agent.
 */
class FALCON_DYN_CLASS CMMsg {
public:
   typedef enum {
      e_resource_signaled,
      e_incoming_context,
      e_context_terminated
   }
   t_type;

   /** Type of message */
   t_type m_type;
   /** data traveling with the message */
   union {
      VMContext* ctx;
      Shared* res;
   }
   m_data;

   CMMsg() {};

   CMMsg( VMContext* ctx ):
      m_type(e_incoming_context) { m_data.ctx = ctx; }

   CMMsg( VMContext* ctx, bool ):
      m_type(e_context_terminated) { m_data.ctx = ctx; }

   CMMsg( Shared* res ):
      m_type(e_resource_signaled) { m_data.res = res; }
};


class ContextManager::Private
{
public:
   typedef std::set<VMContext*> ContextSet;
   typedef std::multimap<int64, VMContext*> ScheduleMap;


   //========================================
   // messages for the scheduler.
   typedef SyncQueue<CMMsg> MsgQueue;
   MsgQueue m_messages;

   ScheduleMap m_schedMap;

   virtual ~Private()
   {
   }

};


ContextManager::ContextManager()
{
   TRACE("ContextManager being created at %p", this );

   m_bStopped = true;
   _p = new Private;
}


ContextManager::~ContextManager()
{
   MESSAGE( "ContextManager destructor start" );
   stop();

   MESSAGE( "ContextManager -- remove reference on runnable contexts" );
   VMContext* ctx = 0;
   while ( m_readyContexts.getST( ctx ) )
   {
      ctx->decref();
   }

   // remove reference on sleeping contexts
   MESSAGE( "ContextManager -- remove reference on sleeping contexts" );
   {
      Private::ScheduleMap::iterator iter = _p->m_schedMap.begin();
      Private::ScheduleMap::iterator end = _p->m_schedMap.end();
      while ( iter != end )
      {
         ctx = iter->second;
         ctx->decref();
         ++iter;
      }
   }

   MESSAGE( "ContextManager -- remove reference on traveling contexts" );
   CMMsg msg;
   while ( _p->m_messages.getST(msg) )
   {
      switch( msg.m_type )
      {
      case CMMsg::e_incoming_context:
      case CMMsg::e_context_terminated:
         msg.m_data.ctx->decref();
         break;

      case CMMsg::e_resource_signaled:
         msg.m_data.res->decref();
         break;

      default:
         // nothing to do
         break;
      }
   }

   delete _p;

   MESSAGE( "ContextManager destructor complete" );
}


bool ContextManager::stop()
{
   MESSAGE( "ContextManager stop begin" );
   m_mtxStopped.lock();
   if(m_bStopped) {
      m_mtxStopped.unlock();
      MESSAGE( "ContextManager already stopped" );
      return false;
   }
   m_bStopped = true;
   m_mtxStopped.unlock();

   // send a killer message
   _p->m_messages.terminateWaiters();

   // Todo: we should also wait for all the processor to be terminated.
   void* result = 0;
   m_thread->join(result);

   MESSAGE( "ContextManager stop complete" );
   return true;

}


bool ContextManager::start()
{
   MESSAGE( "ContextManager start begin" );
   m_mtxStopped.lock();
   if(!m_bStopped) {
      m_mtxStopped.unlock();
      MESSAGE( "ContextManager already started begin" );
      return false;
   }
   m_bStopped = false;
   m_mtxStopped.unlock();

   m_thread = new SysThread( this );
   m_thread->start();

   MESSAGE( "ContextManager start complete" );
   return true;
}


void ContextManager::onContextDescheduled( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextDescheduled %d", ctx->id() );
   ctx->incref();
   // perform automatic release of aqcuired resources
   ctx->acquire(0);
   _p->m_messages.add( CMMsg(ctx) );
}

void ContextManager::onContextTerminated( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextTerminated %d", ctx->id() );
   // perform automatic release of aqcuired resources
   ctx->acquire(0);
   ctx->incref();
   _p->m_messages.add( CMMsg(ctx, true) );
}

void ContextManager::onSharedSignaled( Shared* waitable )
{
   TRACE1( "ContextManager::onSharedSignaled %p", waitable );
   waitable->incref();
   _p->m_messages.add( CMMsg(waitable) );
}


int64 ContextManager::msToAbs( int32 to )
{
   return Sys::_milliseconds() + to;
}


//===========================================================
// Manager
//

void* ContextManager::run()
{
   MESSAGE( "ContextManager::Manager::run -- start");

   m_next_schedule = 0;
   m_now = Sys::_milliseconds();

   int term = 0;
   while( true )
   {
      // How much should be wait?
      int32 timeout = m_next_schedule - m_now;

      // see if we have some message to manage.
      CMMsg msg;
      bool recvd = _p->m_messages.getTimed( msg, timeout, &term );

      // are we done?
      if( term ) {
         break;
      }

      m_now = Sys::_milliseconds();
      manageSleepingContexts();

      if( !recvd ) {
         continue;
      }

      // we never get here if the message list is empty -- and if unlocked

      switch( msg.m_type )
      {
      case CMMsg::e_context_terminated:
         manageTerminatedContext( msg.m_data.ctx );
         msg.m_data.ctx->decref();
         break;

      case CMMsg::e_incoming_context:
         manageDesceduledContext( msg.m_data.ctx );
         msg.m_data.ctx->decref();
         break;

      case CMMsg::e_resource_signaled:
         manageSignal( msg.m_data.res );
         msg.m_data.res->decref();
         break;
      }
   }

   MESSAGE( "ContextManager::Manager::run -- end");
   return 0;
}


void ContextManager::manageReadyContext( VMContext* ctx )
{
   TRACE( "manageReadyContext - Waking context %p(%d)", ctx, ctx->id() );

   // remove the context from any shared resource in wait.
   ctx->abortWaits();

   // ask the context group if the context is quiescent
   if( ctx->inGroup() != 0 )
   {
      bool proceed = ctx->inGroup()->onContextReady(ctx);
      TRACE1( "manageReadyContext - Group says context %p(%d) is %s", ctx, ctx->id(),
               (proceed ? "ready" : "quiescent") );
      if( ! proceed ) {
         return;
      }
   }

   m_readyContexts.add( ctx );
}


bool ContextManager::manageSleepingContexts()
{
   ContextManager::Private::ScheduleMap& mmap = _p->m_schedMap;
   TRACE( "manageSleepingContexts - Checking %d contexts", (int) mmap.size() );
   bool done = false;

   // by default, we don't have a next schedule.
   m_next_schedule = 0;
   while( ! mmap.empty() )
   {
      ContextManager::Private::ScheduleMap::iterator front = mmap.begin();
      int64 sched = front->first;
      if( sched <= m_now ){
         manageReadyContext( front->second );
         mmap.erase(front);
         done = true;
      }
      else {
         // first element scheduled in future.
         m_next_schedule = sched;
         break;
      }
   }

   TRACE( "manageSleepingContexts - Completed with %s",
            (done? "some wakeups" : "no wakeup"));
   return done;
}


void ContextManager::manageTerminatedContext( VMContext* ctx )
{
   TRACE( "manageTerminatedContext - Terminating context %p(%d)", ctx, ctx->id() );
   if( ctx->nextSchedule() != 0 )
   {
      removeSleepingContext( ctx );
   }
}


void ContextManager::removeSleepingContext( VMContext* ctx )
{
   TRACE( "removeSleepingContext - Removing context %p(%d)", ctx, ctx->id() );

   Private::ScheduleMap::iterator pos = _p->m_schedMap.find( ctx->nextSchedule() );
   while( pos != _p->m_schedMap.end() && pos->first == ctx->nextSchedule() ) {
      if (pos->second == ctx)  {
         TRACE( "removeSleepingContext - Context %p(%d) found in wait", ctx, ctx->id() );
         _p->m_schedMap.erase(pos);
         ctx->decref();
         break;
      }
      ++pos;
   }
}

void ContextManager::manageDesceduledContext( VMContext* ctx )
{
   TRACE( "manageDesceduledContext - Managing descheduled context %p(%d)", ctx, ctx->id() );

   // a new context is de-scheduled.
   // TODO: send it to GC if needed.

   // Check if a shared resource was readied during the idle time.
   Shared* sh = ctx->checkAcquiredWait();
   if( sh != 0 ) {
      TRACE( "manageDesceduledContext - Context %p(%d) ready because acquired success", ctx, ctx->id() );
      manageReadyContext( ctx );
   }
   else if( ctx->nextSchedule() >= 0 && ctx->nextSchedule() <= m_now ) {
      // immediately ready
      TRACE( "manageDesceduledContext - Context %p(%d) expire time before now.", ctx, ctx->id() );
      manageReadyContext( ctx );
   }
   else {
      TRACE( "manageDesceduledContext - context %p(%d) put in wait", ctx, ctx->id() );
      _p->m_schedMap.insert( std::make_pair(ctx->nextSchedule(), ctx) );
   }
   // in every case, we keep it.
   ctx->incref();
}


void ContextManager::manageSignal( Shared* shared )
{

   std::deque<VMContext*> readyCtx;
   TRACE("ContextManager::manageSignal -- managing signaled resource %p", shared);

   shared->_p->m_mtx.lock();
   int& signals = shared->_p->m_signals;
   Shared::Private::ContextList::iterator iter = shared->_p->m_waiters.begin();
   Shared::Private::ContextList::iterator end = shared->_p->m_waiters.end();

   while( signals > 0 && iter != end ) {
      VMContext* waiter = *iter;
      readyCtx.push_back( waiter );
      --signals;
      ++iter;
   }
   shared->_p->m_mtx.unlock();

   TRACE("ContextManager::manageSignal -- waking up %d contexts", (int) readyCtx.size() );
   std::deque<VMContext*>::iterator ri = readyCtx.begin();
   std::deque<VMContext*>::iterator re = readyCtx.end();
   if( shared->hasAcquireSemantic() )
   {
      while( ri != re ) {
         VMContext* ctx = *ri;
         ctx->acquire( shared );
         removeSleepingContext(ctx);
         manageReadyContext( ctx );
         ++ri;
      }
   }
   else {
      while( ri != re ) {
         VMContext* ctx = *ri;
         removeSleepingContext(ctx);
         manageReadyContext( ctx );
         ++ri;
      }
   }
}

}

/* end of contextmanager.cpp */