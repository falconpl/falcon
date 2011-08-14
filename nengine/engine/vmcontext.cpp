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
#include <falcon/codeerror.h>

#include <stdlib.h>
#include <string.h>


namespace Falcon {

VMContext::VMContext( VMachine* vm ):
   m_vm(vm),
   m_ruleEntryResult(false)
{
   m_codeStack = (CodeFrame *) malloc(INITIAL_STACK_ALLOC*sizeof(CodeFrame));
   m_topCode = m_codeStack-1;
   m_maxCode = m_codeStack + INITIAL_STACK_ALLOC;

   m_callStack = (CallFrame*)  malloc(INITIAL_STACK_ALLOC*sizeof(CallFrame));
   m_topCall = m_callStack-1;
   m_maxCall = m_callStack + INITIAL_STACK_ALLOC;

   m_dataStack = (Item*) malloc(INITIAL_STACK_ALLOC*sizeof(Item));
   // the data stack can NEVER be empty. -- an empty data stack is an error.
   m_topData = m_dataStack;
   m_maxData = m_dataStack + INITIAL_STACK_ALLOC;

   m_deepStep = 0;

   // prepare a low-limit VM terminator request.
   pushReturn();
}


VMContext::VMContext( bool ):
   m_vm(0),
   m_ruleEntryResult(false)
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
   while ( m_topData + localCount + 1 > m_maxData )
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
   m_topData += localCount; // point to the last local
}


void VMContext::addRuleNDFrame( uint32 tbPoint )
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   int32 oldRuleTop = m_dataStack[stackBase-1].content.mth.ruleTop;

   long localCount = ((m_topData+1) - m_dataStack) - stackBase;
   while ( m_topData + localCount + 1 > m_maxData )
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
   m_topData += localCount;
}


void VMContext::commitRule()
{
   CallFrame& cf = currentFrame();
   long localCount = localVarCount();
   int32 baseRuleTop = params()[-1].content.mth.ruleTop;

   // copy the local variables.
   memcpy( m_dataStack + baseRuleTop, m_dataStack + cf.m_stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = baseRuleTop;
   m_topData = m_dataStack + baseRuleTop + localCount - 1;
}



template<class __checker>
void VMContext::unrollToNext( const __checker& check )
{
   // first, we must have at least a function around.
   register CallFrame* curFrame =  m_topCall;
   register CodeFrame* curCode = m_topCode;
   register Item* curData = m_topData;
   while( m_callStack <= curFrame )
   {
      // then, get the current topCall pointer to code stack.
      CodeFrame* baseCode = m_codeStack + curFrame->m_codeBase;
      // now unroll up to when we're able to hit a next base
      while( curCode >= baseCode )
      {
         if( check( *curCode->m_step ) )
         {
            m_topCall = curFrame;
            m_topData = curData;
            m_topCode = curCode;
            return;
         }
         --curCode;
      }
      // unroll the call.
      curData = m_dataStack + curFrame->m_initBase;
      curFrame--;
   }
   // we didn't find it.
   throw new CodeError( ErrorParam(e_break_out, __LINE__, SRC )
      .origin(ErrorParam::e_orig_vm));
}

class CheckIfCodeIsNextBase
{
public:
   inline bool operator()( const PStep& ps ) const { return ps.isNextBase(); }
};

class CheckIfCodeIsLoopBase
{
public:
   inline bool operator()( const PStep& ps ) const { return ps.isLoopBase(); }
};


void VMContext::unrollToNextBase()
{
   CheckIfCodeIsNextBase checker;
   unrollToNext( checker );
}


void VMContext::unrollToLoopBase()
{
   CheckIfCodeIsLoopBase checker;
   unrollToNext( checker );
}


//===================================================================
// Higher level management
//


void VMContext::pushQuit()
{
   class QuitStep: public PStep {
   public:
      QuitStep() { apply = apply_; }
      virtual ~QuitStep() {}

      inline virtual void describeTo( String& s ) const { s= "#Quit"; }
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


void VMContext::pushComplete()
{
   class CompleteStep: public PStep {
   public:
      CompleteStep() { apply = apply_; }
      virtual ~CompleteStep() {}

      inline virtual void describeTo( String& s ) const { s= "#Complete"; }
   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->vm()->quit();
      }
   };

   static CompleteStep qs;
   pushCode( &qs );
}


void VMContext::pushReturn()
{
   class Step: public PStep {
   public:
      Step() { apply = apply_; }
      virtual ~Step() {}

      inline virtual void describeTo( String& s ) const { s= "#Return"; }
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

      inline virtual void describeTo( String& s ) const { s= "#Break"; }
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


void VMContext::call( Function* function, int nparams, const Item& self )
{
   TRACE( "Calling method %s.%s -- call frame code:%p, data:%p, call:%p",
         self.describe(3).c_ize(), function->locate().c_ize(),
         m_topCode, m_topData, m_topCall  );

   makeCallFrame( function, nparams, self );
   TRACE1( "-- codebase:%d, stackBase:%d, self: %s ", \
         m_topCall->m_codeBase, m_topCall->m_stackBase, self.isNil() ? "nil" : "value"  );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::call( Function* function, int nparams )
{
   TRACE( "Calling function %s -- call frame code:%p, data:%p, call:%p",
         function->locate().c_ize(),m_topCode, m_topData, m_topCall  );

   makeCallFrame( function, nparams );
   TRACE3( "-- codebase:%d, stackBase:%d ", \
         m_topCall->m_codeBase, m_topCall->m_stackBase );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::callItem( const Item& item, int pcount, Item const* params )
{
   TRACE( "Calling item: %s -- call frame code:%p, data:%p, call:%p",
      item.describe(2).c_ize(), m_topCode, m_topData, m_topCall  );

   Class* cls;
   void* data;
   item.forceClassInst( cls, data );
   
   addSpace( pcount+1 );
   *(m_topData - ( pcount + 1 )) = item;
   if( pcount > 0 )
   {      
      memcpy( m_topData-pcount, params, pcount * sizeof(item) );
   }
   
   cls->op_call( this, pcount, data );
}


void VMContext::insertData(int32 pos, Item* data, int32 dataSize, int32 replSize )
{
   addSpace( dataSize - replSize );
   // this is the first item we have to mangle with.
   Item* base = m_topData - (dataSize - replSize + pos-1);
   
   if( pos > replSize )
   {  
      memmove( base + dataSize,
               base + replSize, 
               sizeof(Item) * (pos-replSize) );
   }
   
   memcpy( base, data, sizeof(Item)*dataSize );
}


void VMContext::returnFrame( const Item& value )
{
   register CallFrame* topCall = m_topCall;

   TRACE1( "Return frame from function %s", topCall->m_function->name().c_ize() );

   // reset code and data
   m_topCode = m_codeStack + topCall->m_codeBase-1;
   PARANOID( "Code stack underflow at return", (m_topCode >= m_codeStack-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...

   m_topData = m_dataStack + topCall->m_initBase-1;
   // notice: data stack can never be empty, an empty data stack is an errro.
   PARANOID( "Data stack underflow at return", (m_topData >= m_dataStack) );

   *m_topData = value;

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
