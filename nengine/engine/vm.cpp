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

#include <falcon/stream.h>
#include <falcon/stdstreams.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/transcoder.h>


#include <falcon/genericerror.h>
#include <falcon/locationinfo.h>
#include <falcon/module.h>

#include <falcon/trace.h>
#include <falcon/autocstring.h>

namespace Falcon
{

VMachine::VMachine( Stream* stdIn, Stream* stdOut, Stream* stdErr ):
   m_event(eventNone)
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   m_context = new VMContext;

   if ( stdIn == 0 )
   {
      TRACE1( "Virtual machine create -- loading duplicate standard input stream", 0 );
      m_stdIn = new StdInStream( true );
   }
   else
   {
      m_stdIn = stdIn;
   }

   if ( stdOut == 0 )
   {
      TRACE1( "Virtual machine create -- loading duplicate standard output stream", 0 );
      m_stdOut = new StdOutStream( true );
   }
   else
   {
      m_stdOut = stdOut;
   }

   if ( stdErr == 0 )
   {
      TRACE1( "Virtual machine create -- loading duplicate standard error stream", 0 );
      m_stdErr= new StdErrStream( true );
   }
   else
   {
      m_stdErr = stdErr;
   }

   // TODO Determine system transcoder
   m_stdCoder = Engine::instance()->getTranscoder("C");
   fassert( m_stdCoder != 0 );
   m_bOwnCoder = false;

   m_textIn = new TextReader( m_stdIn, m_stdCoder );
   m_textOut = new TextWriter( m_stdOut, m_stdCoder );
   m_textErr = new TextWriter( m_stdErr, m_stdCoder );

#ifdef FALCON_SYSTEM_WIN
   m_textOut->setCRLF(true);
   m_textErr->setCRLF(true);
#endif

   m_textOut->lineFlush(true);
   m_textErr->lineFlush(true);
}

VMachine::~VMachine()
{
   TRACE( "Virtual machine being destroyed at %p", this );

   delete m_textIn;
   delete m_textOut;
   delete m_textErr;

   delete m_stdIn;
   delete m_stdOut;
   delete m_stdErr;

   if( m_bOwnCoder )
   {
      delete m_stdCoder;
   }

   TRACE( "Virtual machine destroyed at %p", this );
}

void VMachine::stdIn( Stream* s )
{
   delete m_stdIn;
   m_stdIn = s;
   m_textIn->changeStream( s );
}

void VMachine::stdOut( Stream* s )
{
   delete m_stdOut;
   m_stdOut = s;
   m_textOut->changeStream( s );
}

void VMachine::stdErr( Stream* s )
{
   delete m_stdErr;
   m_stdErr = s;
   m_textErr->changeStream( s );
}


bool VMachine::setStdEncoding( const String& name )
{
   Transcoder* tc = Engine::instance()->getTranscoder(name);
   if( tc == 0 )
   {
      return false;
   }
   m_stdCoder = tc;
   m_bOwnCoder = false;

   m_textIn->setEncoding( tc );
   m_textOut->setEncoding( tc );
   m_textErr->setEncoding( tc );
   return true;
}


void VMachine::setStdEncoding( Transcoder* ts, bool bOwn )
{
   m_stdCoder = ts;
   m_bOwnCoder = bOwn;

   m_textIn->setEncoding( ts );
   m_textOut->setEncoding( ts );
   m_textErr->setEncoding( ts );
}


void VMachine::ifDeep( const PStep* postcall )
{
   fassert( m_context->m_deepStep == 0 );
   m_context->m_deepStep = postcall;
}

void VMachine::goingDeep()
{
   if( m_context->m_deepStep )
   {
      currentContext()->pushCode( currentContext()->m_deepStep );
      m_context->m_deepStep = 0;
   }
}


bool VMachine::wentDeep()
{
   bool bWent = m_context->m_deepStep == 0;
   m_context->m_deepStep = 0;
   return bWent;
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

void VMachine::raiseError( Error* e )
{
   e->scriptize(regA());
   m_event = eventRaise;
}

bool VMachine::run()
{
   TRACE( "Run called", 0 );
   m_event = eventNone;
   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );

   while( true )
   {
      // the code should never be empty.
      // to stop the VM the caller should push a terminator, at least.
      fassert( ! codeEmpty() );

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
   TRACE( "Entering function: %s", function->locate().c_ize() );
   
   register VMContext* ctx = m_context;
   TRACE( "-- call frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );

#ifdef NDEBUG
   ctx->makeCallFrame( function, nparams, self );
#else
   CallFrame* topCall = ctx->makeCallFrame( function, nparams, self );
   TRACE1( "-- codebase:%d, stackBase:%d, self: %s ", \
         topCall->m_codeBase, topCall->m_stackBase, self.isNil() ? "nil" : "value"  );
#endif
   
   // prepare for a return that won't touch regA
   ctx->m_regA.setNil();
   
   // do the call
   function->apply( this, nparams );
}


void VMachine::returnFrame()
{
   register VMContext* ctx = m_context;
   register CallFrame* topCall = ctx->m_topCall;

   TRACE1( "Return frame from function %s", AutoCString(topCall->m_function->name()).c_str() );
   
   // set determinism context
   if( ! topCall->m_function->isDeterm() )
   {
      ctx->SetNDContext();
   }

   // reset code and data
   ctx->m_topCode = ctx->m_codeStack + topCall->m_codeBase-1;
   PARANOID( "Code stack underflow at return", (ctx->m_topCode >= ctx->m_codeStack-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...
   ctx->m_topData = ctx->m_dataStack + topCall->m_initBase;
   PARANOID( "Data stack underflow at return", (ctx->m_topData >= ctx->m_dataStack-1) );

   // ... so that we can fill the stack with the function result.
   // -- we always have at least 1 element, that is the function item.
   TRACE1( "-- Adding A register to stack", 1 );
   *ctx->m_topData = ctx->m_regA;

   // Return.
   if( ctx->m_topCall-- ==  ctx->m_callStack )
   {
      // was this the topmost frame?
      m_event = eventComplete;
      TRACE( "Returned from last frame -- declaring complete.", 0 );
   }

   PARANOID( "Call stack underflow at return", (ctx->m_topCall >= ctx->m_callStack-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", ctx->m_topCode, ctx->m_topData, ctx->m_topCall  );
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
      ctx->topData().describe(tmp);
      data += " (" + tmp + ")";
   }

   data.A("; A: ");
   ctx->m_regA.describe(tmp);
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
   //TODO
   return 0;
}

void VMachine::pushQuit()
{
   class QuitStep: public PStep {
   public:
      QuitStep() { apply = apply_; }
      virtual ~QuitStep() {}

   private:
      static void apply_( const PStep* ps, VMachine *vm )
      {
         vm->currentContext()->popCode();
         vm->quit();
      }
   };

   static QuitStep qs;
   currentContext()->pushCode( &qs );
}


void VMachine::pushReturn()
{
   class Step: public PStep {
   public:
      Step() { apply = apply_; }
      virtual ~Step() {}

   private:
      static void apply_( const PStep* ps, VMachine *vm )
      {
         vm->currentContext()->popCode();
         vm->setReturn();
      }
   };

   static Step qs;
   currentContext()->pushCode( &qs );
}

void VMachine::pushBreak()
{
   class Step: public PStep {
   public:
      Step() { apply = apply_; }
      virtual ~Step() {}

   private:
      static void apply_( const PStep* ps, VMachine *vm )
      {
         vm->currentContext()->popCode();
         vm->breakpoint();
      }
   };

   static Step qs;
   currentContext()->pushCode( &qs );
}

}

/* end of vm.cpp */
