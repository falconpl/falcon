/*
   FALCON - The Falcon Programming Language.
   FILE: process.h

   Falcon virtual machine -- process entity.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/process.cpp"

#include <falcon/process.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/mt.h>

#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/closure.h>
#include <falcon/modspace.h>
#include <falcon/synfunc.h>
#include <falcon/error.h>


namespace Falcon {

Process::Process( VMachine* owner ):
   m_vm(owner),
   m_context( new VMContext( this ) ),
   m_event( true, false ),
   m_running(false),
   m_ctxId(0),
   m_error(0),
   m_added(false)
{
   // get an ID for this process.
   m_id = m_vm->getNextProcessID();
   m_context = new VMContext(this, 0);
   m_entry = 0;
}

Process::Process( VMachine* owner, bool bAdded ):
   m_vm(owner),
   m_context( new VMContext( this ) ),
   m_event( true, false ),
   m_running(false),
   m_ctxId(0),
   m_error(0),
   m_added(bAdded)
{
   // get an ID for this process.
   m_id = m_vm->getNextProcessID();
   m_context = new VMContext(this, 0);
   m_entry = 0;
}



Process::~Process() {
   m_context->decref();
   if( m_error != 0 ) {
      m_error->decref();
   }
   delete m_entry;
}


SynFunc* Process::readyEntry()
{
   m_context->reset();
   delete m_entry;
   m_entry = new SynFunc("#Entry");
   m_context->callInternal( m_entry, 0 );

   return m_entry;
}


bool Process::start()
{
   if (! checkRunning() ) {
      return false;
   }

   launch();
   return true;
}

bool Process::start( Function* main, int pcount )
{
   if (! checkRunning() ) {
      return false;
   }

   //Put a VM termination request here.
   m_context->callInternal(main, pcount);
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}

bool Process::start( Closure* main, int pcount )
{
   if (! checkRunning() ) {
      return false;
   }

   //Put a VM termination request here.
   m_context->callInternal(main, pcount);
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}

bool Process::startItem( Item& main, int pcount, Item* params )
{
   if (! checkRunning() ) {
      return false;
   }

   // reset the context prior invoking the entry point
   m_context->callItem(main, pcount, params);
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}


const Item& Process::result() const
{
   return m_context->topData();
}

Item& Process::result()
{
   return m_context->topData();
}


InterruptibleEvent::wait_result_t Process::wait( int32 timeout )
{
   InterruptibleEvent::wait_result_t retval = m_event.wait(timeout);

   if( m_error != 0 ) {
      Error* e = m_error;
      e->incref();
      throw e;
   }

   return retval;
}

void Process::interrupt()
{
   m_event.interrupt();
}

void Process::completed()
{
   m_event.set();
}

void Process::completedWithError( Error* error )
{
   if( m_error !=0 ) {
      m_error->decref();
   }
   m_error = error;
   error->incref();

   m_event.set();
}

void Process::launch()
{
   // The add will eventually launch the process.
   m_vm->addProcess( this, true );
}


void Process::addReadyContext( VMContext* ctx ) {
   ctx->incref();
   m_vm->contextManager().readyContexts().add( ctx );
}

bool Process::checkRunning()
{
   m_mtxRunning.lock();

   if( m_running ) {
      m_mtxRunning.unlock();
      return false;
   }

   m_running = true;
   m_mtxRunning.unlock();

   return true;
}

int32 Process::getNextContextID()
{
   return atomicInc(m_ctxId);
}


}

/* end of process.h */
