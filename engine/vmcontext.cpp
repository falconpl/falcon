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
#include <falcon/stderrors.h>  
#include <falcon/syntree.h>       // for catch -- error check
#include <falcon/symbol.h>    
#include <falcon/dynsymbol.h>

#include <falcon/module.h>       // For getDynSymbolValue
#include <falcon/modspace.h>

#include <falcon/errors/codeerror.h>

#include <falcon/psteps/stmttry.h>      // for catch.


#include <stdlib.h>
#include <string.h>


namespace Falcon {

template<class datatype__>
void VMContext::LinearStack<datatype__>::init( int base )
{
   m_base = (datatype__*) malloc( INITIAL_STACK_ALLOC * sizeof(datatype__) );
   m_top = m_base + base;
   m_max = m_base + INITIAL_STACK_ALLOC;
}

template<class datatype__>
VMContext::LinearStack<datatype__>::~LinearStack()
{
   if( m_base != 0 ) free( m_base );
}

template<class datatype__>
void VMContext::LinearStack<datatype__>::more()
{
   long distance = (long)(m_top - m_base);
   long newSize = (long)(m_max - m_base + INCREMENT_STACK_ALLOC);
   TRACE("Reallocating %p: %d -> %ld", m_base, (int)(m_max - m_base), newSize );

   m_base = (datatype__*) realloc( m_base, newSize * sizeof(datatype__) );
   m_top = m_base + distance;
   m_max = m_base + newSize;
}


//========================================================
//


VMContext::VMContext( VMachine* vm ):
   m_safeCode(0),
   m_thrown(0),
   m_finMode( e_fin_none ),
   m_vm(vm),
   m_ruleEntryResult(false),
   m_catchBlock(0),
   m_event(eventNone)
{
   // prepare a low-limit VM terminator request.
   m_dynsStack.init();
   m_codeStack.init();
   m_callStack.init();
   m_dataStack.init(0);

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
   
   m_dynsStack.reset();
   m_codeStack.reset();
   m_callStack.reset();
   m_dataStack.reset(0);

   // prepare a low-limit VM terminator request.
   pushReturn();
}


void VMContext::setSafeCode()
{
   m_safeCode = m_codeStack.m_top - m_codeStack.m_base;  
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

   memcpy( target, m_dataStack.m_base + start, sizeof(Item) * count );
}


void VMContext::startRuleFrame()
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   long localCount = (long)((m_dataStack.m_top+1) - m_dataStack.m_base) - stackBase;
   while ( m_dataStack.m_top + localCount + 1 > m_dataStack.m_max )
   {
      m_dataStack.more();
   }

   Item& ruleFrame = addDataSlot();
   ruleFrame.type( FLC_ITEM_FRAMING );
   ruleFrame.content.data.val64 = stackBase;
   ruleFrame.content.data.val64 <<= 32;
   ruleFrame.content.data.val64 |= 0xFFFFFFFF;
   ruleFrame.content.mth.ruleTop = stackBase;

   // copy the local variables.
   memcpy( m_dataStack.m_top + 1, m_dataStack.m_base + stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = dataSize();
   m_dataStack.m_top += localCount; // point to the last local
}


void VMContext::addRuleNDFrame( uint32 tbPoint )
{
   CallFrame& cf = currentFrame();
   int32 stackBase = cf.m_stackBase;
   int32 oldRuleTop = m_dataStack.m_base[stackBase-1].content.mth.ruleTop;

   long localCount = (long)((m_dataStack.m_top+1) - m_dataStack.m_base) - stackBase;
   while ( m_dataStack.m_top + localCount + 1 > m_dataStack.m_max )
   {
      m_dataStack.more();
   }

   Item& ruleFrame = addDataSlot();
   ruleFrame.type( FLC_ITEM_FRAMING );
   ruleFrame.content.data.val64 = stackBase;
   ruleFrame.content.data.val64 <<= 32;
   ruleFrame.content.data.val64 |= tbPoint;
   ruleFrame.content.mth.ruleTop = oldRuleTop;

   // copy the local variables.
   memcpy( m_dataStack.m_top + 1, m_dataStack.m_base + stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = dataSize();
   m_dataStack.m_top += localCount;
}


void VMContext::commitRule()
{
   CallFrame& cf = currentFrame();
   long localCount = localVarCount();
   int32 baseRuleTop = params()[-1].content.mth.ruleTop;

   // copy the local variables.
   memcpy( m_dataStack.m_base + baseRuleTop, m_dataStack.m_base + cf.m_stackBase, localCount * sizeof(Item) );

   // move forward the stack base.
   cf.m_stackBase = baseRuleTop;
   m_dataStack.m_top = m_dataStack.m_base + baseRuleTop + localCount - 1;
}



template<class _checker>
bool VMContext::unrollToNext( const _checker& check )
{
   // first, we must have at least a function around.
   register CallFrame* curFrame =  m_callStack.m_top;
   register CodeFrame* curCode = m_codeStack.m_top;
   register Item* curData = m_dataStack.m_top;
   register DynsData* curDyns = m_dynsStack.m_top;
   
   while( m_callStack.m_base <= curFrame )
   {
      // then, get the current topCall pointer to code stack.
      CodeFrame* baseCode = m_codeStack.m_base + curFrame->m_codeBase;
      // now unroll up to when we're able to hit a next base
      while( curCode >= baseCode )
      {
         if( check( *curCode->m_step, this ) )
         {
            m_callStack.m_top = curFrame;
            m_dataStack.m_top = curData;
            m_codeStack.m_top = curCode;
            m_dynsStack.m_top = curDyns;
            return true;
         }
         --curCode;
      }
      
      // were we searching a return?
      if( check.isReturn() ) return true;
      
      // unroll the call.
      curData = m_dataStack.m_base + curFrame->m_initBase;
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
   
   m_codeStack.m_top = m_codeStack.m_base + m_safeCode;
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
         Symbol* sym = m_catchBlock->target();
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
         if( m_catchBlock->target() != 0 )
         {
            Item* value = m_catchBlock->target()->value(this);
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
         m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   makeCallFrame( function, nparams, self );
   TRACE1( "-- codebase:%d, stackBase:%d, self: %s ", \
         m_callStack.m_top->m_codeBase, m_callStack.m_top->m_stackBase, self.isNil() ? "nil" : "value"  );

   // do the call
   function->invoke( this, nparams );
}

void VMContext::call( Function* function, int nparams )
{
   TRACE( "Calling function %s -- call frame code:%p, data:%p, call:%p",
         function->locate().c_ize(),m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   makeCallFrame( function, nparams );
   TRACE3( "-- codebase:%d, stackBase:%d ", \
         m_callStack.m_top->m_codeBase, m_callStack.m_top->m_stackBase );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::call( Function* function, ItemArray* closedData, int nparams )
{
   TRACE( "Calling function %s -- call frame code:%p, data:%p, call:%p",
         function->locate().c_ize(),m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   makeCallFrame( function, closedData, nparams );
   TRACE3( "-- codebase:%d, stackBase:%d ", \
         m_callStack.m_top->m_codeBase, m_callStack.m_top->m_stackBase );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::callItem( const Item& item, int pcount, Item const* params )
{
   TRACE( "Calling item: %s -- call frame code:%p, data:%p, call:%p",
      item.describe(2).c_ize(), m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   Class* cls;
   void* data;
   item.forceClassInst( cls, data );
   
   addSpace( pcount+1 );
   *(m_dataStack.m_top - ( pcount + 1 )) = item;
   if( pcount > 0 )
   {      
      memcpy( m_dataStack.m_top-pcount, params, pcount * sizeof(item) );
   }
   
   cls->op_call( this, pcount, data );
}


void VMContext::insertData(int32 pos, Item* data, int32 dataSize, int32 replSize )
{
   addSpace( dataSize - replSize );
   // this is the first item we have to mangle with.
   Item* base = m_dataStack.m_top - (dataSize - replSize + pos-1);
   
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
   register CallFrame* topCall = m_callStack.m_top;
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
   m_codeStack.unroll( topCall->m_codeBase );
   PARANOID( "Code stack underflow at return", (m_codeStack.m_top >= m_codeStack.m_base-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...

   m_dataStack.unroll( topCall->m_initBase );
   // notice: data stack can never be empty, an empty data stack is an error.
   PARANOID( "Data stack underflow at return", (m_dataStack.m_top >= m_dataStack.m_base) );

   m_dynsStack.unroll( topCall->m_dynsBase );
   PARANOID( "Dynamic Symbols stack underflow at return", (m_dynsStack.m_top >= m_dynsStack.m_base-1) );
   
   // Forward the return value
   *m_dataStack.m_top = value;
   
   // Finalize return -- pop call frame
   // TODO: This is useful only in the interactive mode. Maybe we can use a 
   // specific returnFrame for the interactive mode to achieve this.
   if( m_callStack.m_top-- ==  m_callStack.m_base )
   {
      setComplete();
      MESSAGE( "Returned from last frame -- declaring complete." );
   }

   PARANOID( "Call stack underflow at return", (m_callStack.m_top >= m_callStack.m_base-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );
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


void VMContext::setDynSymbolValue( DynSymbol* dyns, const Item& value )
{
   *getDynSymbolValue(dyns) = value;
}

Item* VMContext::getDynSymbolValue( DynSymbol* dyns )
{
   // search for the dynsymbol in the current context.
   const CallFrame* cf = &currentFrame();
   register DynsData* dd = m_dynsStack.m_top;
   register DynsData* base = m_dynsStack.offset( cf->m_dynsBase );
   while( dd >= base ) {
      if ( dyns == dd->m_sym )
      {
         // Found!
         return dd->m_item.dereference();
      }
   }
   
   // no luck. Descend the frames.
   --cf;
   while( cf >= m_callStack.m_base )
   {
      dd = m_dynsStack.m_top;
      base = m_dynsStack.offset( cf->m_dynsBase );
      while( dd >= base ) {
         if ( dyns == dd->m_sym )
         {
            // Found!
            return dd->m_item.dereference();
         }
      }
      
      // no luck, try with locals.
      fassert( cf->m_function != 0 );
      Symbol* locsym = cf->m_function->symbols().findSymbol( dyns->name() );
      if( locsym != 0 )
      {
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // reference the target local variable into our slot.
         ItemReference::create( 
            m_dataStack.m_base[cf->m_initBase + locsym->id()], 
            newData->m_item );
         return newData->m_item.dereference();
      }
      --cf;
   }
   
   // no luck? Try with module globals.
   Module* master = currentFrame().m_function->module();
   if( master != 0 )
   {
      Symbol* globsym = master->getGlobal( dyns->name() );
      if( globsym != 0 )
      {
         Item* value = globsym->value(this);
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // reference the target local variable into our slot.
         ItemReference::create( 
            *value, 
            newData->m_item );
         return newData->m_item.dereference();
      }
   }
   
   // still no luck? -- what about exporeted symbols in VM?
   ModSpace* ms = vm()->modSpace();
   if( ms != 0 )
   {
      Symbol* expsym = ms->findExportedSymbol( dyns->name() );
      if( expsym != 0 )
      {
         Item* value = expsym->value(this);
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // reference the target local variable into our slot.
         ItemReference::create( 
            *value, 
            newData->m_item );
         return newData->m_item.dereference();
      }
   }
   
   // No luck at all.
   DynsData* newData = m_dynsStack.addSlot();
   newData->m_sym = dyns;
   return newData->m_item.dereference();
}

}

/* end of vmcontext.cpp */
