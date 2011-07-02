/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/itemid.h>
#include <falcon/function.h>
#include <falcon/vm.h>

#include <stdlib.h>
#include <string.h>


namespace Falcon {

VMContext::VMContext( VMachine* vm ):
   m_vm(vm)
{
   m_codeStack = (CodeFrame *) malloc(INITIAL_STACK_ALLOC*sizeof(CodeFrame));
   m_topCode = m_codeStack-1;
   m_maxCode = m_codeStack + INITIAL_STACK_ALLOC;

   m_callStack = (CallFrame*)  malloc(INITIAL_STACK_ALLOC*sizeof(CallFrame));
   m_topCall = m_callStack-1;
   m_maxCall = m_callStack + INITIAL_STACK_ALLOC;

   m_dataStack = (Item*) malloc(INITIAL_STACK_ALLOC*sizeof(Item));
   m_topData = m_dataStack-1;
   m_maxData = m_dataStack + INITIAL_STACK_ALLOC;

   m_deepStep = 0;
}


VMContext::VMContext( bool ):
   m_vm(0)
{
}

VMContext::~VMContext()
{
   free(m_codeStack);
   free(m_callStack);
   free(m_dataStack);
}


void VMContext::moreData()
{
   long distance = m_topData - m_dataStack;
   long newSize = m_maxData - m_dataStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_dataStack, (int)(m_maxData - m_dataStack), newSize );

   m_dataStack = (Item*) realloc( m_dataStack, newSize * sizeof(Item) );
   m_topData = m_dataStack + distance;
   m_maxData = m_dataStack + newSize;
}


void VMContext::copyData( Item* target, size_t count, size_t start)
{
   size_t depth = dataSize();

   if ( start == (size_t)-1)
   {
      start = depth < count ? 0 : depth - count;
   }

   if ( count + start > depth )
   {
      count = depth - start;
   }

   memcpy( target, m_dataStack + start, sizeof(Item) * count );
}


void VMContext::moreCode()
{
   long distance = m_topCode - m_codeStack; // we don't want the size of the code,

   long newSize = m_maxCode - m_codeStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_codeStack, (int)(m_maxCode - m_codeStack), newSize );

   m_codeStack = (CodeFrame*) realloc( m_codeStack, newSize * sizeof(CodeFrame) );
   m_topCode = m_codeStack + distance;
   m_maxCode = m_codeStack + newSize;
}


void VMContext::moreCall()
{
   long distance = m_topCall - m_callStack;
   long newSize = m_maxCall - m_callStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_callStack, (int)(m_maxCall - m_callStack), newSize );

   m_callStack = (CallFrame*) realloc( m_callStack, newSize * sizeof(CallFrame) );
   m_topCall = m_callStack + distance;
   m_maxCall = m_callStack + newSize;
}


void VMContext::startRuleFrame()
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   long localCount = ((m_topData+1) - m_dataStack) - stackBase;
   while ( m_maxData - m_topData < localCount + 1 )
   {
      moreData();
   }

   Item& ruleFrame = addDataSlot();
   ruleFrame.type( FLC_ITEM_FRAMING );
   ruleFrame.content.data.val64 = stackBase;
   ruleFrame.content.data.val64 <<= 32;
   ruleFrame.content.data.val64 |= 0xFFFFFFFF;
   ruleFrame.content.mth.ruleTop = stackBase;

   // copy the local variables.
   memcpy( m_topData + 1, m_dataStack + stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = dataSize();
   m_topData = m_dataStack + cf.m_stackBase + localCount-1;
}


void VMContext::addRuleNDFrame( uint32 tbPoint )
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   int32 oldRuleTop = param(-1)->content.mth.ruleTop;

   long localCount = ((m_topData+1) - m_dataStack) - stackBase;
   while ( m_maxData - m_topData < localCount + 1 )
   {
      moreData();
   }

   Item& ruleFrame = addDataSlot();
   ruleFrame.type( FLC_ITEM_FRAMING );
   ruleFrame.content.data.val64 = stackBase;
   ruleFrame.content.data.val64 <<= 32;
   ruleFrame.content.data.val64 |= tbPoint;
   ruleFrame.content.mth.ruleTop = oldRuleTop;

   // copy the local variables.
   memcpy( m_topData + 1, m_dataStack + stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = dataSize();
   m_topData = m_dataStack + cf.m_stackBase + localCount-1;
}


void VMContext::commitRule()
{
   CallFrame& cf = currentFrame();
   long localCount = localVarCount();
   int32 baseRuleTop = param(-1)->content.mth.ruleTop;

   // copy the local variables.
   memcpy( m_dataStack + baseRuleTop, m_dataStack + cf.m_stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = baseRuleTop;
   m_topData = m_dataStack + baseRuleTop + localCount - 1;
}


//===================================================================
// Higher level management
//

void VMContext::ifDeep( const PStep* postcall )
{
   fassert( m_deepStep == 0 );
   m_deepStep = postcall;
}


void VMContext::goingDeep()
{
   if( m_deepStep )
   {
      pushCode( m_deepStep );
      m_deepStep = 0;
   }
}


bool VMContext::wentDeep()
{
   bool bWent = m_deepStep == 0;
   m_deepStep = 0;
   return bWent;
}


void VMContext::pushQuit()
{
   class QuitStep: public PStep {
   public:
      QuitStep() { apply = apply_; }
      virtual ~QuitStep() {}

   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->vm()->quit();
      }
   };

   static QuitStep qs;
   pushCode( &qs );
}


void VMContext::pushReturn()
{
   class Step: public PStep {
   public:
      Step() { apply = apply_; }
      virtual ~Step() {}

   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->vm()->setReturn();
      }
   };

   static Step qs;
   pushCode( &qs );
}


void VMContext::pushBreak()
{
   class Step: public PStep {
   public:
      Step() { apply = apply_; }
      virtual ~Step() {}

   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->vm()->breakpoint();
      }
   };

   static Step qs;
   pushCode( &qs );
}


void VMContext::call( Function* function, int nparams, const Item& self, bool isExpr )
{
   TRACE( "Entering function: %s", function->locate().c_ize() );

   TRACE( "-- call frame code:%p, data:%p, call:%p", m_topCode, m_topData, m_topCall  );

   CallFrame* topCall = makeCallFrame( function, nparams, self, isExpr );
   TRACE1( "-- codebase:%d, stackBase:%d, self: %s ", \
         topCall->m_codeBase, topCall->m_stackBase, self.isNil() ? "nil" : "value"  );

   topCall->m_bExpression = isExpr;
   // prepare for a return that won't touch regA
   m_regA.setNil();

   // do the call
   function->apply( this, nparams );
}


void VMContext::call( Function* function, int nparams, bool isExpr )
{
   TRACE( "Entering function: %s", function->locate().c_ize() );

   TRACE( "-- call frame code:%p, data:%p, call:%p", m_topCode, m_topData, m_topCall  );

   CallFrame* topCall = makeCallFrame( function, nparams, isExpr );
   TRACE1( "-- codebase:%d, stackBase:%d ", \
         topCall->m_codeBase, topCall->m_stackBase );

   // prepare for a return that won't touch regA
   m_regA.setNil();

   // do the call
   function->apply( this, nparams );
}


void VMContext::returnFrame()
{
   register CallFrame* topCall = m_topCall;

   TRACE1( "Return frame from function %s", topCall->m_function->name().c_ize() );

   // set determinism context
   if( ! topCall->m_function->isDeterm() )
   {
      SetNDContext();
   }

   // reset code and data
   m_topCode = m_codeStack + topCall->m_codeBase-1;
   PARANOID( "Code stack underflow at return", (m_topCode >= m_codeStack-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...

   m_topData = m_dataStack + topCall->m_initBase-1;
   PARANOID( "Data stack underflow at return", (m_topData >= m_dataStack-1) );

   // ... so that we can fill the stack with the function result.
   // -- in expressions we always have at least 1 element, that is the function item.
   if( topCall->m_bExpression )
   {
      MESSAGE1( "-- Adding A register to stack");
      *m_topData = m_regA;
   }
   // Return.
   if( m_topCall-- ==  m_callStack )
   {
      //TODO: -- this is a context problem, not a VM problem.
      // was this the topmost frame?
      m_vm->setComplete();
      MESSAGE( "Returned from last frame -- declaring complete." );
   }

   PARANOID( "Call stack underflow at return", (m_topCall >= m_callStack-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", m_topCode, m_topData, m_topCall  );
}

}

/* end of vmcontext.cpp */
