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

#include <falcon/processor.h>
#include <falcon/paranoid.h>
#include <falcon/trace.h>
#include <falcon/mt.h>

namespace Falcon {

// we can't have an init fiasco as m_me is used only after starting some processor thread.
ThreadSpecific Processor::m_me;

Processor::Processor( int32 id, VMachine* owner ):
      m_id(id),
      m_owner( owner )
{}


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
      delete m_thread;
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
      throw e;
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

   TRACE("Processor %p (id %d) starting", this, this->id() );
   Scheduler::Private* p = m_owner->_p;

   VMContext* ctx;
   while( true ) {
      VMContext* ctx = m_owner->getNextReadyContext();
      if( ctx == 0 ) {
         break;
      }

      m_currentContext = ctx;
      // proceed running with this context
      TRACE("Processor %p (id %d) running context %p(%d)",
               this, this->id(), ctx, ctx->id() );
      execute( ctx );
   }

   return 0;
}


void Processor::manageEvents( VMContext* ctx, int32 events )
{
   if( (events & VMContext::evtSwap) ) {
      if ( ctx->nextSchedule() >= 0 ) {
         TRACE( "Scheduler::Processor::execute processor %p(%d) descheduled context %p(%d) for a while",
                  this, this->id(), ctx, ctx->id() );
         m_owner->putInWait( ctx, ctx->nextSchedule() );
      }
      else {
         TRACE( "Scheduler::Processor::execute processor %p(%d) descheduled forever context %p(%d)",
                  this, this->id(), ctx, ctx->id() );
         m_owner->putInWait(ctx);
      }
   }

   if( (events & VMContext::evtBreak) ) {
      TRACE( "Hit breakpoint before %s ", ctx->location().c_ize() );
   }

   if( (events & VMContext::evtComplete) ) {
      TRACE( "Code completion before %s ", ctx->location().c_ize() );
      ctx->terminated();
   }

   if( (events & VMContext::evtTerminate) ) {
      TRACE( "Termination requset before %s ", ctx->location().c_ize() );
      ctx->terminated();
   }

   if( (events & VMContext::evtRaise) ) {
      Error* e = ctx->detachThrownError();
      onError(e);
   }
}


void Processor::execute( VMContext* ctx )
{
   TRACE( "Scheduler::Processor::execute", (int) ctx->callDepth() );
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
         ctx->raiseError( e );
      }

      register int32 events;
      if( (events = ctx->events()) != 0 )
      {
         manageEvents( ctx, events );
         break;
      }
      // END STEP
   }
}

bool Processor::step()
{
   if( m_currentContext == 0 ) {
      m_currentContext = m_owner->getNextReadyContext();
      if( m_currentContext == 0 ) {
         return false;
      }
   }

   register VMContext* ctx = currentContext();
   TRACE( "Scheduler::Processor::step", (int) ctx->callDepth() );
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

   register int32 events;
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

}
/* end of processor.cpp */
