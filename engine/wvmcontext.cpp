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
   WVMContext* wctx = static_cast<WVMContext*>(ctx);

   ctx->popCode();  // not really necessary, but...

   // save the result -- and remove it from stack.
   wctx->m_result = wctx->topData();
   wctx->popData();

   // and swap out the context from the processor.
   ctx->swapOut(); // remove from process + exit from processor
}


//============================================================================
// Main WVMContext
//============================================================================

WVMContext::WVMContext( Process* prc, ContextGroup* grp ):
     VMContext( prc, grp ),
     m_completeCbFunc(0),
     m_completeData(0)
{
   m_evtComplete = new Event(false, false);
   m_stepComplete = new PStepComplete;
}

WVMContext::~WVMContext()
{
   delete m_evtComplete;
   delete m_stepComplete;
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
   pushCode(m_stepComplete);
   call(f,np,params);
   process()->startContext(this);
}

void WVMContext::start( Closure* closure, int32 np, Item const* params )
{
   m_evtComplete->reset();
   reset();

   pushCode(m_stepComplete);
   call(closure,np,params);
   process()->startContext(this);
}

void WVMContext::startItem( const Item& item, int32 np, Item const* params )
{
   reset();
   pushCode(m_stepComplete);
   callItem(item,np,params);
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

}

/* end of wvmcontext.cpp */
