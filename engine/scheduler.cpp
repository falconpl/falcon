/*
   FALCON - The Falcon Programming Language.
   FILE: scheduler.cpp

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define src "engine/scheduler.cpp"

#include <falcon/scheduler.h>
#include <falcon/mt.h>
#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/sys.h>

#include <map>
#include <set>
#include <vector>

namespace Falcon {

class Scheduler::Private
{
public:
   typedef std::set<VMContext*> ContextSet;
   typedef std::deque<VMContext*> ContextList;
   typedef std::deque<Manager::Msg> ManagerMsgList;
   typedef std::vector<Processor*> ProcessorVector;

   typedef std::multimap<int64, VMContext*> ScheduleMap;

   Mutex m_mtxContexts;
   Mutex m_mtxReadyContexts;
   Event m_evtWorkForProcessors;

   /**
    * Set of all contexts.
    */
   ContextSet m_contexts;

   /**
    * Context ready to be scheduled.
    */
   ContextList m_readyContexts;

   /**
    * Processors used by this scheduler.
    */
   ProcessorVector m_processors;

   /**
    Mutex protecting the message list.
    */
   Mutex m_mtxManMsg;

   /**
    Event signaling there's something new to do for the manager.
    */
   Event m_evtWorkForMan;

   /**
    List of messages going to the manager.
    */
   ManagerMsgList m_manmsg;

   Manager* m_manager;

   bool m_terminate;

   ScheduleMap m_schedMap;

   Private():
      m_evtWorkForProcessors( false, false ),
      m_terminate( true )
   {}

   virtual ~Private()
   {

   }

};

Scheduler::Scheduler()
{
   TRACE("Scheduler being created at %p", this );

   _p = new Private;
   _p->m_manager = new Manager;

}


Scheduler::~Scheduler()
{
   MESSAGE( "Scheduler destructor start" );
   stop();

   // delete all the threads
   Private::ProcessorVector::iterator iter = _p->m_processors.begin();
   while( iter != _p->m_processors.end() ) {
      Processor* proc = *iter;
      delete proc;
      ++iter;
   }

   // stop the manager
   _p->m_mtxManMsg.lock();
   _p->m_manmsg.push_back(Manager::Msg( Manager::Msg::e_stop ));
   _p->m_mtxManMsg.unlock();
   _p->m_evtWorkForMan.set();
   _p->m_manager->join();

   delete _p->m_manager;

   MESSAGE( "Scheduler destructor complete" );
}


void Scheduler::stop()
{
   MESSAGE( "Scheduler stop start" );
   _p->m_mtxReadyContexts.lock();
   if( _p->m_terminate  ) {
      _p->m_mtxReadyContexts.unlock();
      return;
   }

   _p->m_terminate = true;
   _p->m_mtxReadyContexts.unlock();

   _p->m_evtWorkForProcessors.set();

   // join all the threads
   Private::ProcessorVector::iterator iter = _p->m_processors.begin();
   while( iter != _p->m_processors.end() ) {
      Processor* proc = *iter;
      proc->join();
      ++iter;
   }

   MESSAGE( "Scheduler stop complete" );
}

void Scheduler::addContext( VMContext *ctx )
{
   TRACE( "Adding context %p(%d)", ctx, ctx->id() );

   _p->m_mtxContexts.lock();
   _p->m_contexts.insert(ctx);
   _p->m_mtxContexts.unlock();

   // put the context as immediately runnable
   putInWait( ctx );
}

void Scheduler::putAtSleep( VMContext* ctx, uint32 timeout )
{
   TRACE( "Putting at sleep context %p(%d) for %d ms", ctx, ctx->id(), timeout );

   _p->m_mtxReadyContexts.lock();
   _p->m_evtWorkForProcessors.set();
   _p->m_mtxReadyContexts.unlock();
}


void Scheduler::putInWait( VMContext* ctx, time_t wakeupTime )
{

}


void Scheduler::putInWait( VMContext* ctx )
{
   Shared** res = ctx->m_acquiring.m_base;
   Shared** top = ctx->m_acquiring.m_top;
   if( res == top )
   {
      TRACE( "Scheduler::putInWait -- rescheduling %p(%d) for immediate execution", ctx, ctx->id() );
      _p->m_mtxReadyContexts.lock();
      _p->m_readyContexts.push_back( ctx );
      _p->m_evtWorkForProcessors.set();
      _p->m_mtxReadyContexts.unlock();
   }
   else {
      m_ctx
   }


   _p->m_mtxReadyContexts.lock();
   _p->m_readyContexts.insert( ctx );
   _p->m_evtWorkForProcessors.set();
   _p->m_mtxReadyContexts.unlock();

}


void Scheduler::terminateContext( VMContext *ctx )
{

}


void Scheduler::signalResource( Shared* waitable )
{

}

void Scheduler::broadcastResource( Shared* waitable )
{

}


void Scheduler::setProcessors( int32 count )
{

}



//===========================================================
// Manager
//

Scheduler::Manager::Manager( Scheduler* owner ):
  m_owner(owner)
{
   m_thread = new SysThread(this);
   m_thread->start();
}

Scheduler::Manager::~Manager()
{
   delete m_thread;
}


void* Scheduler::Manager::run()
{
   MESSAGE( "Scheduler::Manager::run -- start");
   Scheduler::Private* p = m_owner->_p;

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

   MESSAGE( "Scheduler::Manager::run -- end");
   return 0;
}


bool Scheduler::Manager::manageSleepingContexts()
{
   Scheduler::Private::ScheduleMap& mmap = m_owner->_p->m_schedMap;

   // First, identify the contexts to be waken
   std::deque<VMContext*> toWakeup;

   // by default, we don't have a next schedule.
   m_next_schedule = 0;
   while( ! mmap.empty() )
   {
      Scheduler::Private::ScheduleMap::iterator front = mmap.begin();
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
      Scheduler::Private::ContextList& rctx = m_owner->_p->m_readyContexts;
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


void Scheduler::Manager::onTerminateContext( VMContext* ctx )
{

}


void Scheduler::Manager::onSleepContext( VMContext* ctx, int64 absTime )
{

}


void Scheduler::Manager::onWaitContext( VMContext* ctx, int64 absTime )
{

}


void Scheduler::Manager::onSignal( Shared* res )
{

}


void Scheduler::Manager::onAcquirable( Shared* res )
{

}

//===========================================================
// Processor

}

/* end of scheduler.cpp */

