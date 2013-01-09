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
#include <falcon/errors/genericerror.h>
#include <falcon/vm.h>

namespace Falcon {

// we can't have an init fiasco as m_me is used only after starting some processor thread.
ThreadSpecific Processor::m_me;

Processor::Processor( int32 id, VMachine* owner ):
      m_id(id),
      m_owner( owner ),
      m_onTimeSliceExpired( this )
{
}


void Processor::start()
{
   if( m_thread == 0 ) {
      m_thread = new SysThread(this);
      m_thread->start();
   }
}


void Processor::join()
{
   if( m_thread != 0 )
   {
      void* dummy = 0;
      m_thread->join(dummy);
      m_thread = 0;
   }
}

Processor::~Processor()
{
   join();
}

void Processor::onError( Error* e )
{
   if( m_currentContext->inGroup() != 0 )
   {
      m_currentContext->inGroup()->setError( e );
      m_currentContext->terminated();
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
   // for now, just wrap and raise.

   //TODO: extract the error if the item is an instance of error.
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

   TRACE("Processor %p (id %d) starting", this, this->id() );

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

      m_currentContext = ctx;
      // proceed running with this context
      execute( ctx );
   }

   return 0;
}


void Processor::manageEvents( VMContext* ctx, int32 &events )
{
   if( (events & VMContext::evtBreak) ) {
      TRACE( "Hit breakpoint before %s ", ctx->location().c_ize() );
   }

   if( (events & VMContext::evtComplete) ) {
      TRACE( "Code completion of context %d", ctx->id() );
      ctx->terminated();
      m_owner->contextManager().onContextTerminated( ctx );
   }

   if( (events & VMContext::evtTerminate) ) {
      TRACE( "Termination request before %s ", ctx->location().c_ize() );
      ctx->terminated();
      m_owner->contextManager().onContextTerminated( ctx );
   }

   if( (events & VMContext::evtRaise) ) {
      Error* e = ctx->detachThrownError();
      onError(e);
      ctx->clearEvents();
   }

   if( (events & VMContext::evtSwap) )
   {
      ctx->clearEvents();
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


void Processor::execute( VMContext* ctx )
{
   TRACE( "Processor::execute(%d) %d:%d with depth %d",
            this->id(), ctx->process()->id(), ctx->id(), (int) ctx->callDepth() );
   PARANOID( "Call stack empty", (ctx->callDepth() > 0) );

   while( true )
   {
      // BEGIN STEP
      register const PStep* ps = ctx->currentCode().m_step;

      try
      {
         ps->apply( ps, ctx );
      }
      catch( Error* e )
      {
         Engine::instance()->log()->log(Log::fac_engine, Log::lvl_warn, e->describe() );
         ctx->raiseError( e );
      }

      int32 events;
      if( (events = ctx->events()) != 0 )
      {
         manageEvents( ctx, events );
         // did we reset the event?
         if ( events != 0 ) {
            break;
         }

         ctx->clearEvents();
      }
      // END STEP
   }
}


bool Processor::step()
{
   int wasTerminated = 0;
   if( m_currentContext == 0 )
   {
      TRACE("Processor::step %p (id %d) -- loading a new context.", this, this->id() );
      m_owner->contextManager().readyContexts().tryGet( m_currentContext, &wasTerminated );

      if( m_currentContext == 0 )
      {
         TRACE("Processor::step %p (id %d) -- Can't load a new context", this, this->id() );
         return false;
      }
   }

   register VMContext* ctx = currentContext();
   TRACE( "Processor::step %p (id %d) ctx %d:%d with depth %d",
            this, this->id(), ctx->process()->id(), ctx->id(),
            (int) ctx->callDepth() );
   PARANOID( "Call stack empty", (ctx->callDepth() > 0) );

   // BEGIN STEP
   register const PStep* ps = ctx->currentCode().m_step;

   try
   {
      ps->apply( ps, ctx );
   }
   catch( Error* e )
   {
      ctx->raiseError( e );
   }

   int32 events;
   if( (events = ctx->events()) != 0 )
   {
      manageEvents( ctx, events );
   }

   return true;
}


Processor* Processor::currentProcessor()
{
   return (Processor*) m_me.get();
}

void Processor::onTimeSliceExpired()
{
   m_currentContext->setSwapEvent();
}


Processor::OnTimeSliceExpired::OnTimeSliceExpired( Processor* owner ):
         m_owner( owner )
{}

bool Processor::OnTimeSliceExpired::operator()()
{
   m_owner->onTimeSliceExpired();
   return false;
}

}
/* end of processor.cpp */
