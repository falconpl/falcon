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
#define SRC "engine/project.cpp"

#include <falcon/process.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/mt.h>

namespace Falcon {

Process::Process( VMachine* owner ):
   m_vm(owner)
{
   m_context = new VMContext(this, 0);
   m_id = m_vm->getNextProcessID();
}


Process::Process( VMachine* owner, VMContext* mainContext ):
         m_vm(owner),
         m_context(mainContext)
{
   mainContext->incref();
   m_id = m_vm->getNextProcessID();
}

Process::~Process() {
   m_context->decref();
}


void Process::start( Function* main, int pcount )
{
   m_context->call(main, pcount);
}

void Process::start( Closure* main, int pcount )
{
   m_context->call(main, pcount);
}

void Process::startItem( Item& main, int pcount, Item* params )
{
   m_context->callItem(main, pcount, params);
}

const Item& Process::result() const
{
   return m_context->topData();
}

Item& Process::result()
{
   return m_context->topData();
}


bool Process::wait( int64 timeout )
{
   //TODO
   return false;
}

void Process::interrupt()
{
   //TODO
}

}

/* end of process.h */
