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

#include <falcon/genericerror.h>

#include "falcon/locationinfo.h"
#include "falcon/module.h"

namespace Falcon
{

VMachine::VMachine():
   m_event(eventNone)
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   m_context = new VMContext;
}

VMachine::~VMachine()
{
   TRACE( "Virtual machine destroyed at %p", this );
}

void VMachine::onError( Error* e )
{
   // for now, just raise.
   throw e;
}

void VMachine::onRaise( const Item& item )
{
   // for now, just wrap and raise.

   //TODO: extract the error if the item is an instance of error.
   Error* e = new GenericError( ErrorParam(e_uncaught,__LINE__)
         .module("VM") );
   e->raised( item );
   throw e;
}


void VMachine::raiseItem( const Item& item )
{
   regA() = item;
   m_event = eventRaise;
}


bool VMachine::run()
{
   TRACE( "Run called", 0 );
   m_event = eventNone;
   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );

   while( ! codeEmpty() )
   {
      // BEGIN STEP
      const PStep* ps = currentContext()->currentCode().m_step;

      try
      {
         ps->apply( ps, this );
      }
      catch( Error* e )
      {
         onError( e );
         continue;
      }

      switch( m_event )
      {
         case eventNone: break;

         case eventBreak:
            TRACE( "Hit breakpoint before %s ", location().c_ize() );
            return false;

         case eventComplete:
            TRACE( "Run terminated because lower-level complete detected", 0 );
            return true;

         case eventTerminate:
            TRACE( "Terminating on explicit termination request", 0 );
            return true;

         case eventReturn:
            TRACE( "Retnring on setReturn request", 0 );
            m_event = eventNone;
            return false;

         case eventRaise:
            onRaise( regA() );
            // if we're still alive it means the event was correctly handled
            break;
      }
      // END STEP
   }

   TRACE( "Run terminated because of code exaustion", 0 );
   m_event = eventComplete;
   return true;
}


const PStep* VMachine::nextStep() const
{
   TRACE( "Next step", 0 );
   if( codeEmpty() )
   {
      return 0;
   }
   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );


   CodeFrame& cframe = currentContext()->currentCode();
   const PStep* ps = cframe.m_step;

   if( ps->isComposed() )
   {
      const SynTree* st = static_cast<const SynTree*>(ps);
      return st->at(cframe.m_seqId);
   }
   return ps;
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
   // Used by the VM to insert this opcode if needed to exit functions.
   static StmtReturn s_a_return;

   TRACE( "Entering function: %s", function->locate().c_ize() );
   
   register VMContext* ctx = m_context;
   TRACE( "-- call frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );

   // prepare the call frame.
   CallFrame* topCall = ctx->addCallFrame();
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
   PARANOID( "Code stack underflow at return", (ctx->m_topCode >= ctx->m_codeStack-1) );
   ctx->m_topData = ctx->m_dataStack + topCall->m_stackBase-1;
   PARANOID( "Data stack underflow at return", (ctx->m_topData >= ctx->m_dataStack-1) );

   // Return.
   --ctx->m_topCall;
   PARANOID( "Call stack underflow at return", (ctx->m_topCall >= ctx->m_callStack-1) );

   TRACE( "Return frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );

   // if the call was performed by a call expression, our
   // result shall go in the stack.
   if( ctx->m_topCode > ctx->m_codeStack )
   {
      TRACE1( "-- Adding A register to stack", 1 );
      ctx->pushData(ctx->m_regA);
   }
}

String VMachine::report()
{
   register VMContext* ctx = m_context;

   String data = String("Call: ").N( (int32) ctx->callDepth() )
         .A("; Code: ").N((int32)ctx->codeDepth()).A("/").N(ctx->m_topCode->m_seqId)
         .A("; Data: ").N((int32)ctx->dataSize());

   String tmp;

   if( ctx->dataSize() > 0 )
   {
      ctx->topData().toString(tmp);
      data += " (" + tmp + ")";
   }

   data.A("; A: ");
   ctx->m_regA.toString(tmp);
   data += tmp;

   return data;
}

String VMachine::location() const
{
   LocationInfo infos;
   if ( ! location(infos) )
   {
      return "terminated";
   }

   String temp;
   if( infos.m_moduleUri != "" )
   {
      temp = infos.m_moduleUri;
   }
   else
   {
      temp = infos.m_moduleName != "" ? infos.m_moduleName : "<no module>";
   }

   temp += ":" + (infos.m_function == "" ? "<no func>" : infos.m_function);
   if( infos.m_line )
   {
      temp.A(" (").N(infos.m_line);
      if ( infos.m_char )
      {
         temp.A(":").N(infos.m_char);
      }
      temp.A(")");
   }

   return temp;
}


bool VMachine::location( LocationInfo& infos ) const
{
   // location is given by current function and its module plus current source line.
   if( codeEmpty() )
   {
      return false;
   }

   VMContext* vmc = currentContext();

   if( vmc->callDepth() > 0 && vmc->currentFrame().m_function != 0 )
   {
      Function* f = vmc->currentFrame().m_function;
      if ( f->module() != 0 )
      {
         infos.m_moduleName = f->module()->name();
         infos.m_moduleUri = f->module()->uri();
      }
      else
      {
         infos.m_moduleName = "";
         infos.m_moduleUri = "";
      }

      infos.m_function = f->name();
   }
   else 
   {
      infos.m_moduleName = "";
      infos.m_moduleUri = "";
      infos.m_function = "";
   }

   
   const PStep* ps = nextStep();
   if( ps != 0 )
   {
      infos.m_line = ps->line();
      infos.m_char = ps->chr();
   }
   else
   {
      infos.m_line = 0;
      infos.m_char = 0;
   }

   return true;
}

bool VMachine::step()
{
   if ( codeEmpty() )
   {
      TRACE( "Step terminated", 0 );
      return true;
   }

   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );

   // NOTE: This code must be manually coordinated with vm::run()
   // other solutions, as inline() or macros are either unsafe or
   // clumsy.
   
   // In short, each time vm::run is touched, copy here everything between
   // BEGIN OF STEP - END OF STEP

   // BEGIN OF STEP
   const PStep* ps = currentContext()->currentCode().m_step;
   TRACE( "Step at %s", location().c_ize() );  // this is not in VM::Run
   try
   {
      ps->apply( ps, this );
   }
   catch( Error* e )
   {
      onError( e );
      return true;
   }

   switch( m_event )
   {
      case eventNone: break;

      case eventBreak:
         TRACE( "Hit breakpoint before line %s ", location().c_ize() );
         return false;

      case eventComplete:
         TRACE( "Run terminated because lower-level complete detected", 0 );
         return true;

      case eventTerminate:
         TRACE( "Terminating on explicit termination request", 0 );
         return true;

      case eventReturn:
         TRACE( "Retnring on setReturn request", 0 );
         m_event = eventNone;
         return false;

      case eventRaise:
         onRaise( regA() );
         // if we're still alive it means the event was correctly handled
         break;
   }
   // END OF STEP

   return codeEmpty();  // more data waiting ?
}


Item* VMachine::findLocalItem( const String& name )
{
   return 0;
}

}

/* end of vm.cpp */
