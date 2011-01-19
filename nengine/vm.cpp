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
#include <falcon/statement.h>
#include <falcon/item.h>
#include <falcon/function.h>

namespace Falcon
{

static StmtReturn s_a_return;

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

   return true;
}


PStep* VMachine::nextStep() const
{
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
   // prepare the call frame.
   ++m_topCall;
   m_topCall->m_function = function;
   m_topCall->m_codeBase = codeDepth();
   m_topCall->m_stackBase = dataSize()-nparams;
   m_topCall->m_paramCount = nparams;
   m_topCall->m_self = self;

   // fill the parameters
   while( nparams < function->paramCount() )
   {
      (++m_topData)->setNil();
      ++nparams;
   }

   // fill the locals
   int locals = function->varCount() - function->paramCount();
   while( locals > 0 )
   {
      (++m_topData)->setNil();
      --locals;
   }

   // Generate the code
   // must we add a return?
   if( function->syntree().last()->type() != Statement::return_t )
   {
       pushCode( &s_a_return );
   }

   pushCode( &function->syntree() );

   // prepare for a return that won't touch regA
   m_regA.setNil();
}


void VMachine::returnFrame()
{
   // reset code and data
   m_topCode = m_codeStack + m_topCall->m_codeBase-1;
   m_topData = m_dataStack + m_topCall->m_stackBase-1;

   // Return.
   --m_topCall;

   // if the call was performed by a call expression, our
   // result shall go in the stack.
   if( m_topCode > m_codeStack )
   {
      pushData(m_regA);
   }
}


void VMachine::report( String& data )
{
   data = "Function: " + m_topCall->m_function->name() + "\n";
   data += String("Depth: ").N( callDepth() )
         .A("; Code: ").N(codeDepth()).A("/").N(m_topCode->m_seqId)
         .A("; Stack: ").N(dataSize())
         .A("; A: ");
   String tmp;
   m_regA.toString(tmp);
   data += tmp + "\n";
   data += m_topCode->m_step->toString()+"\n";
}

bool VMachine::step()
{
   if ( codeEmpty() )
   {
      return false;
   }

   const PStep* ps = currentCode().m_step;
   ps->apply( ps, this );
   return true;
}


Item* VMachine::findLocalItem( const String& name )
{
   return 0;
}

}

/* end of vm.cpp */
