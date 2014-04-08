/*
   FALCON - The Falcon Programming Language.
   FILE: processor.cpp

   Processor abstraction in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 05 Aug 2012 16:17:38 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/processor.cpp"

#include <falcon/log.h>
#include <falcon/processor.h>
#include <falcon/paranoid.h>
#include <falcon/trace.h>
#include <falcon/mt.h>
#include <falcon/contextmanager.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/vmcontext.h>
#include <falcon/contextgroup.h>
#include <falcon/locationinfo.h>
#include <falcon/stderrors.h>
#include <falcon/scheduler.h>
#include <falcon/gclock.h>

#include <falcon/vm.h>

#include <falcon/trace.h>

#define FALCON_PROCESS_TIMESLICE 100

namespace Falcon {

   /** Persistent data map.
      We have one of these per thread.
   */
class Processor::Private
{
public:
   typedef std::map<String, GCLock*> PDataMap;
   PDataMap* pdata;

   Private()
   {
      pdata = new PDataMap;
   }

   ~Private()
   {
      PDataMap::iterator iter = pdata->begin();

      while( iter != pdata->end() )
      {
         GCLock* gl = iter->second;
         gl->dispose();
         ++iter;
      }

      delete pdata;
   }
};

// Class used to instantiate a execute with no debug checking.
class NoDebug
{
public:
   bool checkDebug( Processor*, VMContext*, const PStep* )
   {
      // does nothing.
      return true;
   }
};


class Debug
{
public:
   bool checkDebug( Processor* prs, VMContext* ctx, const PStep* ps )
   {
      Process* prc = ctx->process();

      // Is there a breakpoint active on this line?
      if( prc->hitBreakpoint(ctx, ps) )
      {
         prc->onBreakpoint(prs,ctx);
         // the breakpoint operation might have changed the current step
      }
      // always force the next pstep to be presented to the processor.
      ctx->setEmergeEvent();

      // should we exit immediately or execute the next pstep?
      return ps == ctx->currentCode().m_step;
   }
};

// we can't have an init fiasco as m_me is used only after starting some processor thread.
ThreadSpecific Processor::m_me;

Processor::Processor( int32 id, VMachine* owner ):
      m_id(id),
      m_owner( owner ),
      m_thread(0),
      m_activity(0)
{
   _p = new Private;
   m_pdata = new PData;
}


Processor::~Processor()
{
   join();
   delete _p;
   delete m_pdata;
}


void Processor::start()
{
   if( m_thread == 0 )
   {
      TRACE( "Processor::start -- Starting processor thread %d(%p)", id(), this);
      m_thread = new SysThread(this);
      m_thread->start();
   }
   else {
      TRACE( "Processor::start -- Invoking start processor thread %d(%p); but already started.", id(), this);
   }
}


void Processor::join()
{
   if( m_thread != 0 )
   {
      TRACE( "Processor::join -- Joining processor thread %d(%p)", id(), this);
      void* dummy = 0;
      m_thread->join(dummy);
      m_thread = 0;
   }
   else {
      TRACE( "Processor::join -- Joining processor thread %d(%p); but not running", id(), this);
   }
}


void Processor::onError( Error* e )
{
   TRACE( "Processor::onError -- %s", e->describe().c_ize() );

   if( m_currentContext->inGroup() != 0 )
   {
      m_currentContext->inGroup()->setError( e );
      m_currentContext->onTerminated();
   }
   else {
      // this is the top context. We're done.
      Engine::instance()->log()->log( Log::fac_engine, Log::lvl_critical, e->describe() );
      //throw e;
      e->decref();
   }
}


void Processor::onRaise( const Item& item )
{
   LocationInfo lci;
   m_currentContext->location(lci);
   Error* e = new GenericError( ErrorParam(e_uncaught, lci.m_line)
         .module(lci.m_moduleUri)
         .symbol(lci.m_function));
   e->raised( item );

   onError( e );
}


void* Processor::run()
{
   m_me.set(this);
   m_currentContext = 0;
   ContextManager::ReadyContextQueue& rctx = m_owner->contextManager().readyContexts();

   TRACE("Processor::run --  %d(%p) starting", this->id(), this );

   int wasTerminated = 0;
   while( true )
   {
      TRACE("Processor::run %p (id %d) waiting for context", this, this->id() );
      VMContext* ctx;
      rctx.get( ctx, &wasTerminated);

      if( wasTerminated != 0 ) {
         TRACE("Processor::run %p (id %d) being terminated", this, this->id() );
         break;
      }

      // declare the context now active.
      ctx->setStatus(VMContext::statusActive);

      // for the scheduler
      ctx->incref();
      m_activity = m_owner->scheduler().addActivity( FALCON_PROCESS_TIMESLICE, onTimesliceExpired, ctx, true );

      // proceed running with this context
      m_currentContext = ctx;

      // select to execute with or without debug checking.
      if( ctx->process()->inDebug() )
      {
         execute<Debug>( ctx );
      }
      else {
         execute<NoDebug>( ctx );
      }

      if( m_owner->scheduler().cancelActivity(m_activity) ) {
         ctx->decref();
      }
      // if false, the contex has been decreffed by onTimesliceExpired
   }

   return 0;
}


void Processor::manageEvents( VMContext* ctx, int32 &events )
{
   ctx->setStatus(VMContext::statusZombie);

   if( (events & VMContext::evtBreak) ) {
      TRACE( "Hit breakpoint before %s ", ctx->location().c_ize() );

      events &= ~VMContext::evtBreak;
      ctx->clearBreakpointEvent();
      ctx->process()->setDebug(true);
      ctx->process()->onBreakpoint(this, ctx);
      events = ctx->events();
   }

   if( (events & VMContext::evtEmerge) ) {
      MESSAGE( "Hit emerge event" );
      events &= ~VMContext::evtEmerge;
   }

   if( (events & VMContext::evtRaise) ) {
      TRACE( "Uncaught error raise in context %d", ctx->id() );
      ctx->onTerminated();
   }
   else if( (events & VMContext::evtComplete) ) {
      TRACE( "Code completion of context %d", ctx->id() );
      ctx->onComplete();
   }
   else if( (events & VMContext::evtTerminate) ) {
      TRACE( "Termination request before %s ", ctx->location().c_ize() );
      ctx->onTerminated();
   }
   else if( (events & VMContext::evtSwap) )
   {
      ctx->clearEvents();
      if( ctx->acquired() != 0 )
      {
         ctx->delayEvents( VMContext::evtSwap );
         events &= ~VMContext::evtSwap;
      }
      else {
         if ( ctx->nextSchedule() >= 0 ) {
            TRACE( "Processor::manageEvents processor %p(%d) descheduled context %p(%d) for a while",
                     this, this->id(), ctx, ctx->id() );
            m_owner->contextManager().onContextDescheduled( ctx );
         }
         else {
            TRACE( "Processor::manageEvents processor %p(%d) descheduled forever context %p(%d)",
                     this, this->id(), ctx, ctx->id() );
            m_owner->contextManager().onContextDescheduled( ctx );
         }
      }
   }
   else if( (events & VMContext::evtTimeslice) )
   {
      ctx->clearEvents();
      events &= ~VMContext::evtTimeslice;

      if( ctx->acquired() != 0 )
      {
         ctx->delayEvents( VMContext::evtTimeslice );
      }
      else
      {
         TRACE( "Processor::manageEvents processor %p(%d) received a timeslice event context %p(%d)",
                  this, this->id(), ctx, ctx->id() );

         // have we other contexts?
         VMContext* newCtx = 0;
         int32 term = 0;
         if( m_owner->contextManager().readyContexts().tryGet( newCtx, &term ) )
         {
            // yep, we're changed.
            m_currentContext = newCtx;
            // we don't hold any extra ref...
            m_owner->contextManager().onContextDescheduled( ctx );

         }

         // need to do reschedule even if not changed.
         if( m_owner->scheduler().cancelActivity( m_activity) )
         {
            // ... except the one for the activity
            ctx->decref();
         }

         m_currentContext->incref();
         m_activity = m_owner->scheduler().addActivity( FALCON_PROCESS_TIMESLICE, onTimesliceExpired, m_currentContext, true );
      }
   }

}


template<class _DCHECK>
   void Processor::execute( VMContext* ctx )
{
   TRACE( "Processor::execute(%d) %d:%d with depth %d",
            this->id(), ctx->process()->id(), ctx->id(), (int) ctx->callDepth() );
   PARANOID( "Call stack empty", (ctx->callDepth() > 0) );

   _DCHECK check;
   ctx->setStatus(VMContext::statusActive);

   while( true )
   {
      // BEGIN STEP
      register const PStep* ps = ctx->currentCode().m_step;

      try
      {
         if(check.checkDebug(this, ctx, ps))
         {
            ps->apply( ps, ctx );
         }
      }
      catch( Error* e )
      {
         ctx->raiseError( e );
      }

      int32 events;
      if( (events = ctx->events()) != 0 )
      {
         manageEvents( ctx, events );
         // did we reset the event?
         if ( events != 0 ) {
            // out of business with this context.
            break;
         }
         // refetch the context, that may be changed.
         ctx = m_currentContext;
         ctx->setStatus(VMContext::statusActive);
      }
      // END STEP
   }
}


void Processor::step( VMContext* ctx )
{
   ctx->setBreakpointEvent();
   execute<NoDebug>(ctx);
   ctx->clearBreakpointEvent();
}


Processor* Processor::currentProcessor()
{
   return (Processor*) m_me.get();
}


void Processor::onTimesliceExpired( void* dt, Scheduler::Activity* )
{
   VMContext* ctx = static_cast<VMContext*>(dt);
   ctx->setTimesliceEvent();
   // we have an extra reference to take care of.
   ctx->decref();
   // we cannot complete the activity, let the main thread try to cancel it.
}


}
/* end of processor.cpp */
