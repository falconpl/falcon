/*
   FALCON - The Falcon Programming Language.
   FILE: vm.cpp

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/symbol.h>
#include <falcon/syntree.h>

namespace Falcon
{

VMachine::VMachine():
      VMContext( false )
{
   // create the first context
   loadContext( new VMContext );
}

VMachine::~VMachine()
{
}


void VMachine::loadContext( VMContext* vmc )
{
   m_context = vmc;

   m_regA = vmc->m_regA;

   m_codeStack = vmc->m_codeStack;
   m_topCode = vmc->m_topCode;

   m_callStack = vmc->m_callStack;
   m_topCall = vmc->m_topCall;

   m_dataStack = vmc->m_dataStack;
   m_topData = vmc->m_topData;
}



void VMachine::saveContext( VMContext* vmc )
{
   vmc->m_regA = vmc->m_regA;
   vmc->m_codeStack = m_codeStack;
   vmc->m_topCode = m_topCode;

   vmc->m_codeStack = m_codeStack;
   vmc->m_topCall = m_topCall;

   vmc->m_dataStack = m_dataStack;
   vmc->m_topData = m_topData;
}


bool VMachine::run()
{
   while( ! codeEmpty() )
   {
      const PStep* ps = currentCode().m_step;
      ps->apply( ps, this );
   }
}


PStep* VMachine::nextStep() const
{
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
}


bool VMachine::step()
{
   if ( codeEmpty() )
   {
      return false;
   }

   const PStep* ps = currentCode().m_step;
   ps->apply( ps, this );
}


void VMachine::returnFrame()
{
}


Item* VMachine::findLocalItem( const String& name )
{
   return 0;
}

}

/* end of vm.cpp */
