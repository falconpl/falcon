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

#define src "engine/contextmanager.cpp"

#include <falcon/contextmanager.h>
#include <falcon/mt.h>
#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/sys.h>
#include <falcon/syncqueue.h>

#include <map>
#include <set>
#include <vector>

#include <syncqueue.h>

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
      e_stop
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

   CMMsg():
      m_type(e_stop) {}

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

   //==============================================
   // Context ready to be scheduled.
   typedef SyncQueue<VMContext*> ContextSyncQueue;
   ContextSyncQueue m_readyContexts;

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

   m_bStopped = false;
   _p = new Private;
}


ContextManager::~ContextManager()
{
   MESSAGE( "ContextManager destructor start" );
   stop();

   MESSAGE( "ContextManager -- remove reference on runnable contexts" );
   VMContext* ctx = 0;
   while ( _p->m_readyContexts.getST( ctx ) )
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
      switch( msg.t_type ) {
      case CMMsg::e_incoming_context:
      case CMMsg::e_context_terminated:
         msg.m_data.ctx->decref();
         break;

      default:
         // nothing to do
         break;
      }
   }

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
   _p->m_messages.add(CMMsg());

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

   m_thread->start();

   MESSAGE( "ContextManager start complete" );
   return true;
}


void ContextManager::onContextDescheduled( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextDescheduled %d", ctx->id() );
   ctx->incref();
   _p->m_messages.add( CMMsg(ctx) );
}

void ContextManager::onContextTerminated( VMContext* ctx )
{
   TRACE1( "ContextManager::onContextTerminated %d", ctx->id() );
   ctx->incref();
   _p->m_messages.add( CMMsg(ctx, true) );
}

void ContextManager::onSharedSignaled( Shared* waitable )
{
   TRACE1( "ContextManager::onSharedSignaled %p", waitable );
   _p->m_messages.add( CMMsg(waitable) );
}


int64 ContextManager::msToAbs( int32 to )
{
   // todo
   return (int64) to;
}


VMContext* ContextManager::getNextReadyContext( int* terminateHandler )
{
   VMContext* ctx = 0;
   if( _p->m_readyContexts.get(ctx, terminateHandler ) ) {
      return ctx;
   }
   return 0;
}


void ContextManager::terminateWaiterForReadyContext( int* terminateHandler )
{
   _p->m_readyContexts.terminateOne( terminateHandler );
}


//===========================================================
// Manager
//

void* ContextManager::run()
{
   MESSAGE( "ContextManager::Manager::run -- start");
   Private* p = _p;

   m_next_schedule = 0;
   m_now = Sys::_milliseconds();

   bool cont = true;
   while( cont )
   {
      // see if we have some message to manage.
      p->m_mtxManMsg.lock();
      if( p->m_manmsg.empty() )
      {
         p->m_mtxManMsg.unlock();

         // NO? -- shall we wait?
         int64 timeout = m_next_schedule - m_now;
         if( timeout > 0 ) {
            p->m_evtWorkForMan.wait(timeout);
         }
         else {
            // wake up sleeping contexts
            if( ! manageSleepingContexts() ) {
               // if we have none, try to wait for a message -- forever.
               p->m_evtWorkForMan.wait();
            }
         }
         m_now = Sys::_milliseconds();

         continue;
      }

      // we never get here if the message list is empty -- and if unlocked
      Msg front( p->m_manmsg.front() );
      p->m_manmsg.pop_front();
      p->m_mtxManMsg.unlock();

      switch( front.m_type ) {
      case Msg::e_terminate_context:
         onTerminateContext( front.m_data.ctx );
         break;

      case Msg::e_sleep_context:
         onSleepContext( front.m_data.ctx, front.m_absTime );
         break;

      case Msg::e_wait_context:
         onWaitContext( front.m_data.ctx, front.m_absTime );
         break;

      case Msg::e_signal_resource:
         onSignal( front.m_data.res );
         break;

      case Msg::e_aquirable_resource:
         onAcquirable( front.m_data.res );
         break;

      case Msg::e_stop:
         cont = false;
         break;
      }
   }

   MESSAGE( "ContextManager::Manager::run -- end");
   return 0;
}


bool ContextManager::Manager::manageSleepingContexts()
{
   ContextManager::Private::ScheduleMap& mmap = m_owner->_p->m_schedMap;

   // First, identify the contexts to be waken
   std::deque<VMContext*> toWakeup;

   // by default, we don't have a next schedule.
   m_next_schedule = 0;
   while( ! mmap.empty() )
   {
      ContextManager::Private::ScheduleMap::iterator front = mmap.begin();
      int64 sched = front->first;
      if( sched <= m_now ){
         toWakeup.push_back(front->second);
         mmap.erase(front)
      }
      else {
         // first element scheduled in future.
         m_next_schedule = sched;
         break;
      }
   }

   // spare a bit of extra calculations if we have nothing to wake up.
   if( ! toWakeup.empty() )
   {
      // now we have to really wake up the contexts in schedule.
      std::deque<VMContext*>::iterator iWakeCtx = toWakeup.begin();

      // -- we'll do in two steps; first retire the waits, then post them for execution.
      while( iWakeCtx != toWakeup.end() )
      {
         VMContext* waken = *iWakeCtx;
         waken->abortWaits();
         // declare the waken context running or waiting to run.
         waken->nextSchedule(-1);
         ++iWakeCtx;
      }

      // time to post the contexts for immediate execution.
      ContextManager::Private::ContextList& rctx = m_owner->_p->m_readyContexts;
      iWakeCtx = toWakeup.begin();
      m_owner->_p->m_mtxReadyContexts.lock();
      while( iWakeCtx != toWakeup.end() ) {
         VMContext* waken = *iWakeCtx;
         rctx.push_back(waken);
         ++iWakeCtx;
      }

      m_owner->_p->m_evtWorkForProcessors.set();
      m_owner->_p->m_mtxReadyContexts.unlock();

      // we did wake up some contexts
      return true;
   }

   // no sleeping contexts in need of wake ups
   return false;
}

}

/* end of contextmanager.cpp */

