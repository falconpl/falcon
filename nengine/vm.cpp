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

#include <falcon/trace.h>

namespace Falcon
{

static StmtReturn s_a_return;

VMachine::VMachine()
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   m_context = new VMContext;
}

VMachine::~VMachine()
{
   TRACE( "Virtual machine destroyed at %p", this );
}

bool VMachine::run()
{
   TRACE( "Run called", 0 );

   //Use the stack, don't use register in this loop
   VMContext* ctx = currentContext();

   while( ! codeEmpty() )
   {
      const PStep* ps = ctx->currentCode().m_step;
      ps->apply( ps, this );
   }

   TRACE( "Run terminated", 0 );
   return true;
}


PStep* VMachine::nextStep() const
{
   TRACE( "Next step", 0 );
   return NULL;
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
   TRACE( "Entering function: %s", function->locate().c_ize() );
   
   register VMContext* ctx = m_context;
   TRACE( "-- call frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );

   // prepare the call frame.
   CallFrame* topCall = ++ctx->m_topCall;
   topCall->m_function = function;
   topCall->m_codeBase = ctx->codeDepth();
   topCall->m_stackBase = ctx->dataSize()-nparams;
   topCall->m_paramCount = nparams;
   topCall->m_self = self;
   TRACE1( "-- codebase:%d, stackBase:%d, self: %s ", \
         topCall->m_codeBase, topCall->m_stackBase, self.isNil() ? "nil" : "value"  );


   // fill the parameters
   TRACE1( "-- filing parameters: %d/%d", nparams, function->paramCount() );
   while( nparams < function->paramCount() )
   {
      (++ctx->m_topData)->setNil();
      ++nparams;
   }

   // fill the locals
   int locals = function->varCount() - function->paramCount();
   TRACE1( "-- filing locals: %d", locals );
   while( locals > 0 )
   {
      (++ctx->m_topData)->setNil();
      --locals;
   }

   // Generate the code
   // must we add a return?
   if( function->syntree().last()->type() != Statement::return_t )
   {
      TRACE1( "-- Pushing extra return", 0 );
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

   TRACE( "Return frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );

   // if the call was performed by a call expression, our
   // result shall go in the stack.
   if( ctx->m_topCode > ctx->m_codeStack )
   {
      TRACE1( "-- Adding A register to stack", 1 );
      ctx->pushData(ctx->m_regA);
   }
}

/*
void VMachine::report( Report& data )
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
*/

bool VMachine::step()
{
   TRACE( "Step", 0 );
   if ( codeEmpty() )
   {
      TRACE( "Step terminated", 0 );
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
