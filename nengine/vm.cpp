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

VMachine::VMachine()
{
   // create the first context
   m_context = new VMContext;
}

VMachine::~VMachine()
{
}

bool VMachine::run()
{
   //Use the stack, don't use register in this loop
   VMContext* ctx = currentContext();

   while( ! codeEmpty() )
   {
      const PStep* ps = ctx->currentCode().m_step;
      ps->apply( ps, this );
   }

   return true;
}


PStep* VMachine::nextStep() const
{
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
   register VMContext* ctx = m_context;

   // prepare the call frame.
   CallFrame* topCall = ++ctx->m_topCall;
   topCall->m_function = function;
   topCall->m_codeBase = ctx->codeDepth();
   topCall->m_stackBase = ctx->dataSize()-nparams;
   topCall->m_paramCount = nparams;
   topCall->m_self = self;

   // fill the parameters

   while( nparams < function->paramCount() )
   {
      (++ctx->m_topData)->setNil();
      ++nparams;
   }

   // fill the locals
   int locals = function->varCount() - function->paramCount();
   while( locals > 0 )
   {
      (++ctx->m_topData)->setNil();
      --locals;
   }

   // Generate the code
   // must we add a return?
   if( function->syntree().last()->type() != Statement::return_t )
   {
       ctx->pushCode( &s_a_return );
   }

   ctx->pushCode( &function->syntree() );

   // prepare for a return that won't touch regA
   ctx->m_regA.setNil();
}


void VMachine::returnFrame()
{
   register VMContext* ctx = m_context;
   CallFrame* topCall = ctx->m_topCall;

   // reset code and data
   ctx->m_topCode = ctx->m_codeStack + topCall->m_codeBase-1;
   ctx->m_topData = ctx->m_dataStack + topCall->m_stackBase-1;

   // Return.
   --ctx->m_topCall;

   // if the call was performed by a call expression, our
   // result shall go in the stack.
   if( ctx->m_topCode > ctx->m_codeStack )
   {
      ctx->pushData(ctx->m_regA);
   }
}


void VMachine::report( String& data )
{
   register VMContext* ctx = m_context;

   data = "Function: " + ctx->m_topCall->m_function->name() + "\n";
   data += String("Depth: ").N( ctx->callDepth() )
         .A("; Code: ").N(ctx->codeDepth()).A("/").N(ctx->m_topCode->m_seqId)
         .A("; Stack: ").N(ctx->dataSize())
         .A("; A: ");
   String tmp;
   ctx->m_regA.toString(tmp);
   data += tmp + "\n";
   data += ctx->m_topCode->m_step->toString()+"\n";
}

bool VMachine::step()
{
   if ( codeEmpty() )
   {
      return false;
   }

   const PStep* ps = m_context->currentCode().m_step;
   ps->apply( ps, this );
   return true;
}


Item* VMachine::findLocalItem( const String& name )
{
   return 0;
}

}

/* end of vm.cpp */
