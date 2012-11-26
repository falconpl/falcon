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


namespace Falcon {

Process::Process( VMachine* owner ):
   m_vm(owner),
   m_context( new VMContext( this ) ),
   m_event( true, false ),
   m_running(false)
{
   m_context = new VMContext(this, 0);
   m_id = m_vm->getNextProcessID();
}


Process::~Process() {
   m_context->decref();
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

   m_context->call(main, pcount);
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

   m_context->call(main, pcount);
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
   return m_event.wait(timeout);
}

void Process::interrupt()
{
   m_event.interrupt();
}

void Process::completed()
{
   m_event.set();
}

void Process::launch()
{
   VMContext* ctx = mainContext();
   m_vm->modSpace()->readyContext( ctx );
   // we're assigning the context to the processor/vm/manager system.
   ctx->incref();
   // processors are synchronized on the context queue.
   m_vm->readyContexts().add( ctx );
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


}

/* end of process.h */
