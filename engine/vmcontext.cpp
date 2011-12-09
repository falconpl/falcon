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
#include <falcon/engine.h>       // for catch -- error check
#include <falcon/stderrors.h>    // for catch -- error check
#include <falcon/syntree.h>    // for catch -- error check
#include <falcon/symbol.h>    // for catch -- error check

#include <falcon/errors/codeerror.h>

#include <falcon/psteps/stmttry.h>      // for catch.


#include <stdlib.h>
#include <string.h>


namespace Falcon {

VMContext::VMContext( VMachine* vm ):
   m_safeCode(0),
   m_thrown(0),
   m_finMode( e_fin_none ),
   m_vm(vm),
   m_ruleEntryResult(false),
   m_catchBlock(0),
   m_event(eventNone)
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

   // prepare a low-limit VM terminator request.
   pushReturn();
}


VMContext::VMContext( bool ):
   m_safeCode(0),
   m_thrown(0),
   m_finMode( e_fin_none ),
   m_vm(0),
   m_ruleEntryResult(false),
   m_catchBlock(0),
   m_event(eventNone)
{
}

VMContext::~VMContext()
{
   free(m_codeStack);
   free(m_callStack);
   free(m_dataStack);
   if( m_thrown != 0 ) m_thrown->decref();   
}


void VMContext::reset()
{
   if( m_thrown != 0 ) m_thrown->decref();
   m_thrown = 0;

   m_event = eventNone;
   m_catchBlock = 0;
   m_ruleEntryResult = false;
   m_finMode = e_fin_none;
   
   m_topCode = m_codeStack-1;
   m_topCall = m_callStack-1;
   // the data stack can NEVER be empty. -- an empty data stack is an error.
   m_topData = m_dataStack;

   // prepare a low-limit VM terminator request.
   pushReturn();
}


void VMContext::setSafeCode()
{
   m_safeCode = m_topCode - m_codeStack;  
}


void VMContext::moreData()
{
   long distance = (long)(m_topData - m_dataStack);
   long newSize = (long)(m_maxData - m_dataStack + INCREMENT_STACK_ALLOC);
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
   long distance = (long)(m_topCode - m_codeStack); // we don't want the size of the code,

   long newSize = (long)(m_maxCode - m_codeStack + INCREMENT_STACK_ALLOC);
   TRACE("Reallocating %p: %d -> %ld", m_codeStack, (int)(m_maxCode - m_codeStack), newSize );

   m_codeStack = (CodeFrame*) realloc( m_codeStack, newSize * sizeof(CodeFrame) );
   m_topCode = m_codeStack + distance;
   m_maxCode = m_codeStack + newSize;
}


void VMContext::moreCall()
{
   long distance = (long)(m_topCall - m_callStack);
   long newSize = (long)(m_maxCall - m_callStack + INCREMENT_STACK_ALLOC);
   TRACE("Reallocating %p: %d -> %ld", m_callStack, (int)(m_maxCall - m_callStack), newSize );

   m_callStack = (CallFrame*) realloc( m_callStack, newSize * sizeof(CallFrame) );
   m_topCall = m_callStack + distance;
   m_maxCall = m_callStack + newSize;
}


void VMContext::startRuleFrame()
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   long localCount = (long)((m_topData+1) - m_dataStack) - stackBase;
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

   long localCount = (long)((m_topData+1) - m_dataStack) - stackBase;
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



template<class _checker>
bool VMContext::unrollToNext( const _checker& check )
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
         if( check( *curCode->m_step, this ) )
         {
            m_topCall = curFrame;
            m_topData = curData;
            m_topCode = curCode;
            return true;
         }
         --curCode;
      }
      
      // were we searching a return?
      if( check.isReturn() ) return true;
      
      // unroll the call.
      curData = m_dataStack + curFrame->m_initBase;
      curFrame--;
   }
  
   return false;
}

class CheckIfCodeIsNextBase
{
public:
   inline bool operator()( const PStep& ps, VMContext* ctx ) const 
   {
      if( ps.isFinally() )
      {
         ctx->setFinallyContinuation( VMContext::e_fin_continue );
         return true;
      }
      
      return ps.isNextBase(); 
   }
   
   bool isReturn() const { return false; }
};

class CheckIfCodeIsLoopBase
{
public:
   inline bool operator()( const PStep& ps, VMContext* ctx ) const 
   { 
      if( ps.isFinally() )
      {
         ctx->setFinallyContinuation( VMContext::e_fin_break );
         return true;
      }
      return ps.isLoopBase(); 
   }
   
   bool isReturn() const { return false; }
};



class CheckIfCodeIsCatchItem
{
public:
   CheckIfCodeIsCatchItem( const Item& item ):
      m_item(item)
   {}
   
   inline bool operator()( const PStep& ps, VMContext* ctx ) const 
   {       
      if( ps.isFinally() )
      {
         ctx->setFinallyContinuation( VMContext::e_fin_raise );
         return true;
      }
      
      if( ps.isCatch() )
      {
         const StmtTry* stry = static_cast<const StmtTry*>( &ps );
         SynTree* st = stry->catchSelect().findBlockForItem( m_item );
         ctx->setCatchBlock( st );
         return st != 0;
      }
      
      return false;
   }
   
   bool isReturn() const { return false; }
   
private:
   const Item& m_item;
};

class CheckIfCodeIsCatchError
{
public:
   CheckIfCodeIsCatchError( Class* err ):
      m_err(err)
   {}
   
   inline bool operator()( const PStep& ps, VMContext* ctx ) const 
   {       
      if( ps.isFinally() )
      {
         ctx->setFinallyContinuation( VMContext::e_fin_raise );
         return true;
      }
      
      if( ps.isCatch() )
      {
         const StmtTry* stry = static_cast<const StmtTry*>( &ps );
         SynTree* st = stry->catchSelect().findBlockForClass( m_err );
         if ( st == 0 ) st = stry->catchSelect().getDefault();
         ctx->setCatchBlock( st );
         return st != 0;
      }
      
      return false;
   }
   
   bool isReturn() const { return false; }
   
private:
   Class* m_err;
};


class CheckIfCodeIsReturn
{
public:
   
   inline bool operator()( const PStep& ps, VMContext* ctx ) const 
   {       
      if( ps.isFinally() )
      {
         ctx->setFinallyContinuation( VMContext::e_fin_return );
         return true;
      }
      
      return false;
   }
   
   bool isReturn() const { return true; }
};


void VMContext::unrollToNextBase()
{
   CheckIfCodeIsNextBase checker;
   if( ! unrollToNext( checker ) )
   {
      raiseError( new CodeError( ErrorParam(e_continue_out, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)) );
   }
}


void VMContext::unrollToLoopBase()
{
   CheckIfCodeIsLoopBase checker;
   if( ! unrollToNext( checker ) )
   {
      raiseError( new CodeError( ErrorParam(e_break_out, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)) );
   }
}

bool VMContext::unrollToSafeCode()
{
   if ( m_safeCode == 0 )
   {
      return false;
   }
   
   m_topCode = m_codeStack + m_safeCode;
   return true;
}

//===================================================================
// Try frame management.
//

void VMContext::raiseItem( const Item& item )
{
   // first, if this is a boxed error, unbox it and send it to raiseError.
   if( item.isUser() )
   {
      if ( item.asClass()->isErrorClass() ) {
         raiseError( static_cast<Error*>( item.asInst() ) );
         return;
      }
   }
   
   // are we in a finally? -- in that case, we must just queue our item.
   if( m_finMode == e_fin_raise && m_thrown != 0 )
   {
      // in this case, the effect is that of continue raisal of an uncaught item.
      CodeError* ce = new CodeError( ErrorParam( e_uncaught, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm));
      ce->raised( item );
      raiseError( ce );
      return;
   }
   
   // ok, it's a real item. Just in case, remove the rising-error marker.
   if( m_thrown != 0 ) m_thrown->decref();
   m_thrown = 0;

   // can we catch it?
   CheckIfCodeIsCatchItem check(item);
   
   m_catchBlock = 0;
   if( unrollToNext<CheckIfCodeIsCatchItem>( check ) )
   {
      // the unroller has prepared the code for us
      if( m_catchBlock != 0 )
      {
         resetCode( m_catchBlock );
         Symbol* sym = m_catchBlock->headSymbol();
         if( sym != 0 )
         {
            *sym->value(this) = item; 
         }
         m_raised.setNil();
      }
      else
      {
         // otherwise, it was a finally, and we should leave it alone.
         // Save the item for later re-raisal.
         m_raised = item;
      }
   }
   else 
   {   
      // no luck. 
      
      // reset the raised object, anyhow.
      m_raised.setNil();
      
      CodeError* ce = new CodeError( ErrorParam( e_uncaught, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm));
      ce->raised( item );
      raiseError( ce );
   }
}
   
void VMContext::raiseError( Error* ce )
{   
   // are we in a finally? 
   // -- in that case, we must queue our error and continue the previous raise
   if( m_finMode == e_fin_raise && m_thrown != 0 )
   {     
      m_thrown->appendSubError( ce );
      ce->decref();
      ce = m_thrown;
   }
   else
   {
      if( m_thrown != 0 ) m_thrown->decref();
      m_thrown = 0;
   }
   
   // can we catch it?   
   m_catchBlock = 0;
   CheckIfCodeIsCatchError check( ce->handler() );
   if( unrollToNext<CheckIfCodeIsCatchError>( check ) )
   {
      // the unroller has prepared the code for us
      if( m_catchBlock != 0 )
      {
         resetCode( m_catchBlock );

         // assign the error to the required item.
         if( m_catchBlock->headSymbol() != 0 )
         {
            Item* value = m_catchBlock->headSymbol()->value(this);
            if( value != 0 )
            {
               value->setUser( ce->handler(), ce, true );
               ce->decref();
            }
         }
      }
      else
      {
         // otherwise, we have a finally around. 
         ce->incref();
         m_thrown = ce;
      }
   }
   else
   {   
      // prevent script-bound re-catching.
      m_thrown = ce;
      m_event = eventRaise;
   }
}


void VMContext::unhandledError( Error* ce )
{
   if( m_thrown != 0 ) m_thrown->decref();
   m_thrown = ce;
   
   m_event = eventRaise;
}


void VMContext::finallyComplete()
{
   /** reduce the count of traversed finally codes. */
   switch( m_finMode )
   {
      case e_fin_none: break;
      case e_fin_raise: 
         if ( m_thrown != 0 ) 
         {
            // if we don't zero the thrown error, the system will be fooled and think
            // that a new throw has appened.
            Error* e = m_thrown;
            m_thrown = 0;
            raiseError( e );
         }
         else
         {
            raiseItem( m_raised );
         }
         break;
         
      case e_fin_break:
         unrollToLoopBase();
         break;
               
      case e_fin_continue:
         unrollToNextBase();
         break;
         
      case e_fin_return:
         {
            Item copy = m_raised;
            m_raised.setNil();
            returnFrame( copy );
         }
         break;
         
      case e_fin_terminate:
         // currently not used.
         break;
   }
   
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
         ctx->quit();
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
         ctx->quit();
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
         ctx->setReturn();
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
         ctx->breakpoint();
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

   if( topCall->m_finallyCount > 0 )
   {
      // we have some finally block in the middle that must be respected.
      CheckIfCodeIsReturn check;
      unrollToNext( check );
      m_raised = value;
      return;
   }
   
   // reset code and data
   m_topCode = m_codeStack + topCall->m_codeBase-1;
   PARANOID( "Code stack underflow at return", (m_topCode >= m_codeStack-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...

   m_topData = m_dataStack + topCall->m_initBase-1;
   // notice: data stack can never be empty, an empty data stack is an errro.
   PARANOID( "Data stack underflow at return", (m_topData >= m_dataStack) );

   // Forward the return value
   *m_topData = value;
   
   // Finalize return -- pop call frame
   if( m_topCall-- ==  m_callStack )
   {
      setComplete();
      MESSAGE( "Returned from last frame -- declaring complete." );
   }

   PARANOID( "Call stack underflow at return", (m_topCall >= m_callStack-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", m_topCode, m_topData, m_topCall  );
}



bool VMContext::boolTopData()
{
   
   switch( topData().type() )
   {
   case FLC_ITEM_NIL:
      return false;

   case FLC_ITEM_BOOL:
      return topData().asBoolean();

   case FLC_ITEM_INT:
      return topData().asInteger() != 0;

   case FLC_ITEM_NUM:
      return topData().asNumeric() != 0.0;

   case FLC_ITEM_USER:
      topData().asClass()->op_isTrue( this, topData().asInst() );
      if(topData().isBoolean() )
      {
         return topData().asBoolean();
      }
   }
   
   return false;
}

}

/* end of vmcontext.cpp */
