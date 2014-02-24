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


#include <stdio.h>

namespace Falcon {

/**
 Message exchanged between the public methods and the manager agent.
 */
class FALCON_DYN_CLASS CMMsg {
public:
   typedef enum {
      e_resource_signaled,
      e_incoming_context,
      e_context_terminated,
      e_group_terminated,
      e_wakeup_context
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

   CMMsg( VMContext* ctx, t_type t ):
         m_type(t) { m_data.ctx = ctx; }

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

void ContextManager::onContextReady( VMContext* ctx )
{
   TRACE( "ContextManager::onContextReady -- ready context %p(%d:%d)", ctx, ctx->process()->id(), ctx->id() );
   readyContexts().add( ctx );
}

void ContextManager::onContextDescheduled( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextDescheduled %d", ctx->id() );
   ctx->incref();
   ctx->setStatus(VMContext::statusDescheduled);
   _p->m_messages.add( CMMsg(ctx) );
}

void ContextManager::onContextTerminated( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextTerminated %p(%d) in process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );
   ctx->incref();
   _p->m_messages.add( CMMsg(ctx, CMMsg::e_context_terminated) );
}

void ContextManager::onGroupTerminated( VMContext* ctx )
{
   TRACE1( "ContextManager::onGroupTerminated %p(%d) in process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );
   ctx->incref();
   _p->m_messages.add( CMMsg(ctx, CMMsg::e_group_terminated) );
}

void ContextManager::onSharedSignaled( Shared* waitable )
{
   TRACE1( "ContextManager::onSharedSignaled %p", waitable );
   waitable->incref();
   _p->m_messages.add( CMMsg(waitable) );
}

void ContextManager::wakeUp( VMContext* ctx )
{
   TRACE1( "ContextManager::wakeUp %p(%d) in process %p(%d)",
            ctx, ctx->id(), ctx->process(), ctx->process()->id() );

   ctx->incref();
   _p->m_messages.add( CMMsg(ctx, CMMsg::e_wakeup_context) );
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
      int64 timeout = m_next_schedule - m_now;

      // see if we have some message to manage.
      CMMsg msg;
      bool recvd;
      if( timeout > 0 ) {
         TRACE1( "ContextManager::Manager::run -- waiting messages for %d msecs", (int) timeout );
         recvd = _p->m_messages.getTimed( msg, (int32) timeout, &term );
      }
      else {
         if( m_next_schedule != 0 ) {
            MESSAGE1( "ContextManager::Manager::run -- schedule expired, checking for new messages" );
            recvd = _p->m_messages.tryGet( msg, &term );
         }
         else {
            MESSAGE1( "ContextManager::Manager::run -- endlessly waiting for new messages" );
            //recvd = _p->m_messages.getTimed( msg, 1000, &term );
            recvd = _p->m_messages.get( msg, &term );
         }
      }

      // are we done?
      if( term ) {
         break;
      }

      m_now = Sys::_milliseconds();

      if( !recvd ) {
         manageSleepingContexts();
         continue;
      }

      // we never get here if the message list is empty -- and if unlocked

      switch( msg.m_type )
      {
      case CMMsg::e_context_terminated:
         manageTerminatedContext( msg.m_data.ctx );
         msg.m_data.ctx->decref();
         break;

      case CMMsg::e_group_terminated:
         manageContextKilledInGroup( msg.m_data.ctx );
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

      case CMMsg::e_wakeup_context:
         manageAwakenContext( msg.m_data.ctx );
         msg.m_data.ctx->decref();
         break;
      }

      manageSleepingContexts();
   }

   MESSAGE( "ContextManager::Manager::run -- end");
   return 0;
}


void ContextManager::manageReadyContext( VMContext* ctx )
{
   TRACE( "manageReadyContext - Waking context %p(%d)", ctx, ctx->id() );

   // remove the context from any shared resource in wait.
   ctx->setStatus(VMContext::statusReady);
   ctx->abortWaits();

   // ask the context group if the context is quiescent
   if( ctx->inGroup() != 0 )
   {
      bool proceed = ctx->inGroup()->onContextReady(ctx);
      TRACE1( "manageReadyContext - Group says context %p(%d) is %s", ctx, ctx->id(),
               (proceed ? "ready" : "quiescent") );
      if( ! proceed ) {
         ctx->setStatus(VMContext::statusQuiescent);
         return;
      }
   }

   TRACE1( "manageReadyContext - Adding ready context %p(%d)", ctx, ctx->id() );
   ctx->nextSchedule(0); // ready to run.
   m_readyContexts.add( ctx );
}


bool ContextManager::manageSleepingContexts()
{
   ContextManager::Private::ScheduleMap& mmap = _p->m_schedMap;
   TRACE( "manageSleepingContexts - Checking %d contexts", (int) mmap.size() );
   bool done = false;

   // by default, we don't have a next schedule.
   m_next_schedule = 0;
   ContextManager::Private::ScheduleMap::iterator front = mmap.begin();
   ContextManager::Private::ScheduleMap::iterator end = mmap.end();

   while( front != end )
   {
      int64 sched = front->first;
      TRACE2( "manageSleepingContexts - schedule %d for %d(%p) in %d(%p)",
               (int) sched, front->second->id(), front->second, front->second->process()->id(), front->second->process() );
      if( sched <= m_now ) {
         if( sched >= 0 )
         {
            manageReadyContext( front->second );
            ContextManager::Private::ScheduleMap::iterator old = front;
            ++front;
            mmap.erase(old);
            done = true;
         }
         else {
            // keep -1 until something in future is found.
            ++front;
         }
      }
      else {
         // first element scheduled in future.
         m_next_schedule = sched;
         break;
      }
   }

   TRACE( "manageSleepingContexts - Completed with %s at %d",
            (done? "some wakeups" : "no wakeup"), (int) m_next_schedule);
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

void ContextManager::manageContextKilledInGroup( VMContext* ctx )
{
   TRACE( "manageContextKilledInGroup - Terminating context %p(%d)", ctx, ctx->id() );
   if( removeSleepingContext( ctx ) )
   {
      // this context is gone now
      ctx->inGroup()->onContextTerminated(ctx);
   }
}

void ContextManager::manageAwakenContext( VMContext* ctx )
{
   TRACE( "manageAwakenContext - Sending the context to the collector. %p(%d)", ctx, ctx->id() );
   // if the schedule is 0, the context is NOT sleeping,
   // but if it's non-zero, it might somewhere being in bring of getting asleep.
   if( ctx->nextSchedule() != 0 && removeSleepingContext( ctx ) )
   {
      if( ctx->markedForInspection() ) {
         Engine::collector()->offerContext(ctx);
         // the collector took it.
         MESSAGE( "ContextManager::manageAwakenContext - accepted by the collector"  );
      }
      else if( ctx->isTerminated() )
      {
         TRACE( "manageAwakenContext - Context %p(%d) terminated.", ctx, ctx->id() );
         ctx->onTerminated();
         ctx->decref();
      }
   }
   else {
      TRACE( "manageAwakenContext - Context %p(%d) was already readied for running.", ctx, ctx->id() );
   }

}

bool ContextManager::removeSleepingContext( VMContext* ctx )
{
   TRACE( "removeSleepingContext - Removing context %p(%d) with sched %d", ctx, ctx->id(), (int) ctx->nextSchedule() );

   Private::ScheduleMap::iterator pos = _p->m_schedMap.find( ctx->nextSchedule() );
   while( pos != _p->m_schedMap.end() && pos->first == ctx->nextSchedule() ) {
      if (pos->second == ctx)  {
         TRACE( "removeSleepingContext - Context %p(%d) found in wait", ctx, ctx->id() );
         _p->m_schedMap.erase(pos);
         ctx->awake();
         ctx->decref();
         return true;
      }
      ++pos;
   }

   return false;
}

void ContextManager::manageDesceduledContext( VMContext* ctx )
{
   TRACE( "manageDesceduledContext - %d(%p) in %d(%p)",
            ctx->id(), ctx, ctx->process()->id(), ctx->process() );

   if( ctx->isTerminated() )
   {
      TRACE( "manageDesceduledContext - Context was terminated prior reaching here %d(%p) in %d(%p)",
               ctx->id(), ctx, ctx->process()->id(), ctx->process() );

      if( ctx->isActive() )
      {
         ctx->onTerminated();
      }
      // we don't keep it.
      return;
   }

   // a new context is de-scheduled.
   if( ctx->markedForInspection() ) {
      Engine::collector()->offerContext(ctx);
      // the collector took it.
      MESSAGE( "ContextManager::manageDesceduledContext - accepted by the collector"  );
      return;
   }

   // Check if a shared resource was readied during the idle time.
   Shared* sh = ctx->declareWaits();
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
      TRACE( "manageDesceduledContext - context %p(%d) put in wait to sched %d", ctx, ctx->id(), (int)ctx->nextSchedule());
      if( ! ctx->goToSleep() ) {
         TRACE( "manageDesceduledContext - context %p(%d) requested urgently by the collector %d", ctx, ctx->id(), (int)ctx->nextSchedule());
         // ops, the context was urgently asked by the collector.
         // a context that can't go to sleep MUST be accepted by the collector.
         Engine::collector()->offerContext(ctx);
         return;
      }
      else {
         if( ctx->waitingSharedCount() ) {
            ctx->setStatus( VMContext::statusWaiting  );
         }
         else {
            ctx->setStatus( VMContext::statusSleeping );
         }
         _p->m_schedMap.insert( std::make_pair(ctx->nextSchedule(), ctx) );
      }
   }
   // in every case, we keep it.
   ctx->incref();
}


void ContextManager::manageSignal( Shared* shared )
{

   std::deque<VMContext*> readyCtx;
   TRACE("ContextManager::manageSignal -- managing signaled resource %p", shared);

   shared->_p->m_mtx.lock();
   Shared::Private::ContextList& clist = shared->_p->m_waiters;
   if (shared->isContetxSpecific())
   {
      Shared::Private::ContextList::iterator cli = clist.begin();

      while( cli != clist.end() )
      {
         if( shared->lockedConsumeSignal( clist.front() ) )
         {
            VMContext* waiter = *cli;
            readyCtx.push_back( waiter );
            waiter->decref();
            cli = clist.erase(cli);
         }
         else {
            ++cli;
         }
      }
   }
   else
   {
      while( ! clist.empty() && shared->lockedConsumeSignal( clist.front() ) ) {
         VMContext* waiter = clist.front();
         readyCtx.push_back( waiter );
         waiter->decref();
         clist.pop_front();
      }
   }

   shared->onWakeupComplete();
   shared->_p->m_mtx.unlock();

   TRACE("ContextManager::manageSignal -- waking up %d contexts", (int) readyCtx.size() );
   std::deque<VMContext*>::iterator ri = readyCtx.begin();
   std::deque<VMContext*>::iterator re = readyCtx.end();
   if( shared->hasAcquireSemantic() )
   {
      while( ri != re )
      {
         VMContext* ctx = *ri;
         ctx->signaledResource( shared );
         ctx->acquire( shared );
         if( removeSleepingContext(ctx) ) {
            manageReadyContext( ctx );
         }
         else {
            ctx->nextSchedule(0); // prevents accepting it if incoming
         }
         // else, the contex is alive or will be.
         ++ri;
      }
   }
   else {
      while( ri != re )
      {
         VMContext* ctx = *ri;
         ctx->signaledResource( shared );
         if( removeSleepingContext(ctx) ) {
            manageReadyContext( ctx );
         }
         else {
            ctx->nextSchedule(0); // prevents accepting it if incoming
         }
         ++ri;
      }
   }
}

}

/* end of contextmanager.cpp */
