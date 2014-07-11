/*
   FALCON - The Falcon Programming Language.
   FILE: wvmcontext.cpp

   Falcon virtual machine -- waitable VM context.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/wvmcontext.cpp"

#include <falcon/wvmcontext.h>
#include <falcon/pstep.h>
#include <falcon/syntree.h>
#include <falcon/psteps/stmttry.h>
#include <falcon/stderrors.h>
#include <falcon/synfunc.h>
#include <falcon/dyncompiler.h>

#include <falcon/stringstream.h>
#include <falcon/textreader.h>

namespace Falcon {

class WVMContext::PStepComplete: public PStep
{
public:
   PStepComplete(){ apply = apply_; }
   virtual ~PStepComplete() {}

private:
   static void apply_(const PStep* ps, VMContext* ctx );
};

void WVMContext::PStepComplete::apply_(const PStep*, VMContext* ctx )
{
   MESSAGE( "WVMContext::PStepComplete::apply_" );
   WVMContext* wctx = static_cast<WVMContext*>(ctx);

   ctx->popCode();  // not really necessary, but...

   // save the result -- and remove it from stack.
   wctx->m_result = wctx->topData();
   wctx->popData();

   // and swap out the context from the processor.
   ctx->swapOut(); // remove from process + exit from processor
}


class WVMContext::PStepErrorGate: public SynTree
{
public:
   PStepErrorGate(){ apply = apply_; }
   virtual ~PStepErrorGate() {}

   void describeTo( String& tgt ) const
   {
      tgt = "WVMContext::PStepErrorGate";
   }

private:
   static void apply_(const PStep* ps, VMContext* ctx );
};


void WVMContext::PStepErrorGate::apply_(const PStep*, VMContext* ctx )
{
   MESSAGE( "WVMContext::PStepErrorGate::apply_" );

   WVMContext* wctx = static_cast<WVMContext*>(ctx);

   ctx->popCode(); // not really necessary, the ctx is going to die, but...

   if( ctx->thrownError() == 0 )
   {
      CodeError* error = FALCON_SIGN_ERROR(CodeError, e_uncaught );
      error->raised( ctx->raised() );
      wctx->completeWithError( error );
      error->decref();
   }
   else {
      wctx->completeWithError(ctx->thrownError());
   }
}

//============================================================================
// Main WVMContext
//============================================================================

WVMContext::WVMContext( Process* prc, ContextGroup* grp ):
     VMContext( prc, grp ),
     m_completeCbFunc(0),
     m_completeData(0),
     m_completionError(0)
{
   // event is hand-reset.
   m_evtComplete = new Event(false, false);

   // create the completion step
   m_stepComplete = new PStepComplete;

   // prepare the error gate.
   StmtTry* errorGate = new StmtTry;
   errorGate->catchSelect().append( new PStepErrorGate );
   m_stepErrorGate = errorGate;

   m_baseFrame = new SynFunc("<base>");
}

WVMContext::~WVMContext()
{
   if( m_completionError != 0 )
   {
      m_completionError->decref();
   }

   delete m_evtComplete;
   delete m_stepComplete;
   delete m_baseFrame;
}


void WVMContext::onComplete()
{
   // invoke the function if necessary
   if( m_completeCbFunc != 0 )
   {
      m_completeCbFunc( this, m_completeData );
   }

   // then set the event...
   m_evtComplete->set();
}


void WVMContext::start( Function* f, int32 np, Item const* params )
{
   m_evtComplete->reset();
   reset();
   call(f,np,params);
   process()->startContext(this);
}

void WVMContext::start( Closure* closure, int32 np, Item const* params )
{
   m_evtComplete->reset();
   reset();
   call(closure,np,params);
   process()->startContext(this);
}

void WVMContext::startItem( const Item& item, int32 np, Item const* params )
{
   m_evtComplete->reset();
   reset();
   callItem(item,np,params);
   process()->startContext(this);
}


void WVMContext::startEvaluation( const String& script )
{
   StringStream* ss = new StringStream(script);
   TextReader tr(ss);
   ss->decref();
   startEvaluation( &tr );
}


void WVMContext::startEvaluation( TextReader* tr )
{
   DynCompiler dc(this);

   reset();

   // in case of throw, we're clean without unclean memory.
   SynTree* st = dc.compile( tr );

   pushData( Item(FALCON_GC_HANDLE(st)) );
   pushCode( st );
   process()->startContext(this);
}


void WVMContext::setOnComplete( complete_cbfunc func, void* data )
{
   m_completeCbFunc = func;
   m_completeData = data;
}

void WVMContext::gcPerformMark()
{
   VMContext::gcPerformMark();
   m_result.gcMark(m_currentMark);
}


bool WVMContext::wait(int32 to) const
{
   bool result = m_evtComplete->wait(to);

   m_mtxCompletionError.lock();
   if ( m_completionError != 0 )
   {
      Error* error = m_completionError;
      m_mtxCompletionError.unlock();
      throw error;
   }
   else {
      m_mtxCompletionError.unlock();
   }

   return result;
}


void WVMContext::completeWithError( Error* error )
{
   error->incref();
   Error* old;

   m_mtxCompletionError.lock();
   old = m_completionError;
   m_completionError = error;
   m_mtxCompletionError.unlock();

   if( old != 0 )
   {
      old->decref();
   }

   swapOut();
}

void WVMContext::reset()
{
   VMContext::reset();
   Error* old;

   m_mtxCompletionError.lock();
   old = m_completionError;
   m_completionError = 0;
   m_mtxCompletionError.unlock();

   if( old != 0 )
   {
      old->decref();
   }

   call(m_baseFrame);
   pushCodeWithUnrollPoint(m_stepErrorGate);
   pushCode(m_stepComplete);
}

}

/* end of wvmcontext.cpp */
