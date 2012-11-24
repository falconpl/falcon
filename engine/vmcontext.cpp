/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Single agent in a virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "engine/vmcontext.cpp"

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
#include <falcon/stdsteps.h>
#include <falcon/shared.h>
#include <falcon/sys.h>
#include <falcon/contextgroup.h>

#include <falcon/module.h>       // For getDynSymbolValue
#include <falcon/modspace.h>

#include <falcon/storer.h>
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
void VMContext::LinearStack<datatype__>::init( int base, uint32 allocSize )
{
   m_base = (datatype__*) malloc( allocSize * sizeof(datatype__) );
   m_top = m_base + base;
   m_max = m_base + allocSize;
   m_allocSize = allocSize;
}

template<class datatype__>
VMContext::LinearStack<datatype__>::~LinearStack()
{
   if( m_base != 0 ) free( m_base );
}

//========================================================
//


VMContext::VMContext( Process* prc, ContextGroup* grp ):
   m_safeCode(0),
   m_thrown(0),
   m_finMode( e_fin_none ),
   m_ruleEntryResult(false),
   m_catchBlock(0),
   m_id(0),
   m_next_schedule(0),
   m_inGroup(grp),
   m_process(prc)
{
   // prepare a low-limit VM terminator request.
   m_dynsStack.init();
   m_codeStack.init();
   m_callStack.init();
   m_locsStack.init();
   m_dataStack.init(0, m_dataStack.INITIAL_STACK_ALLOC*1000);

   m_waiting.init();
   m_acquired = 0;
   atomicSet(m_events,0);
   pushReturn();
}


VMContext::~VMContext()
{
   if( m_thrown != 0 ) m_thrown->decref();
}


void VMContext::reset()
{
   if( m_thrown != 0 ) m_thrown->decref();
   m_thrown = 0;
   atomicSet(m_events, 0);

   // do not reset ingroup.

   m_catchBlock = 0;
   m_ruleEntryResult = false;
   m_finMode = e_fin_none;

   m_dynsStack.reset();
   m_codeStack.reset();
   m_callStack.reset();
   m_locsStack.reset();
   m_dataStack.reset(0);

   abortWaits();
   if( m_acquired != 0 ) {
      m_acquired->signal();
   }
   m_waiting.init();
   m_acquired = 0;

   // prepare a low-limit VM terminator request.
   pushReturn();
}



bool VMContext::location( LocationInfo& infos ) const
{
   // location is given by current function and its module plus current source line.
   if( codeEmpty() )
   {
      return false;
   }

   if( callDepth() > 0 && currentFrame().m_function != 0 )
   {
      Function* f = currentFrame().m_function;
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


String VMContext::location() const
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


String VMContext::report()
{
   register VMContext* ctx = this;

   String data = String("Call: ").N( (int32) ctx->callDepth() )
         .A("; Code: ").N((int32)ctx->codeDepth()).A("/").N(ctx->currentCode().m_seqId)
         .A("; Data: ").N((int32)ctx->dataSize());

   String tmp;

   if( ctx->dataSize() > 0 )
   {
      ctx->topData().describe(tmp);
      data += " (" + tmp + ")";
   }

   data += tmp;

   return data;
}


const PStep* VMContext::nextStep() const
{
   MESSAGE( "Next step" );
   if( codeEmpty() )
   {
      return 0;
   }
   PARANOID( "Call stack empty", (this->callDepth() > 0) );


   const CodeFrame& cframe = this->currentCode();
   const PStep* ps = cframe.m_step;

   if( ps->isComposed() )
   {
      const SynTree* st = static_cast<const SynTree*>(ps);
      return st->at(cframe.m_seqId);
   }
   return ps;
}


 Storer* VMContext::getTopStorer() const
 {
    const CodeFrame* cc = &currentCode();
    // todo; better check
    if( cc != 0  )
    {
       uint32 flags = cc->m_step->flags();
       switch( flags )
       {
          case 1:
          {
             const Storer::WriteNext* wn = static_cast<const Storer::WriteNext*>(cc->m_step);
             return wn->storer();
          }
          break;

          case 2:
          {
             const Storer::TraverseNext* tn = static_cast<const Storer::TraverseNext*>(cc->m_step);
             return tn->storer();
          }
          break;
       }
    }
    
    return 0;
 }

void VMContext::setSafeCode()
{
   m_safeCode = m_codeStack.m_top - m_codeStack.m_base;
}


void VMContext::abortWaits()
{
   Shared** base,** top;

   base = m_waiting.m_base;
   top = m_waiting.m_top;
   while( base != top ) {
      (*base)->dropWaiting( this );
      ++base;
   }
   m_waiting.m_top = m_waiting.m_base;
}


void VMContext::initWait()
{
   m_next_schedule = -1;
   m_waiting.m_top = m_waiting.m_base;
}

void VMContext::addWait( Shared* resource )
{
   *m_waiting.addSlot() = resource;
}

Shared* VMContext::engageWait( int64 timeout )
{
   Shared** base = m_waiting.m_base;
   Shared** top = m_waiting.m_top;
   while( base != top ) {
      if ((*base)->consumeSignal()) {
         m_waiting.m_top = m_waiting.m_base;
         TRACE( "VMContext::engageWait got signaled %p", *base);
         return *base;
      }
      ++base;
   }
   MESSAGE( "VMContext::engageWait nothing signaled, will go wait");

   // nothing free right now, put us at sleep.
   setSwapEvent();
   m_next_schedule = timeout > 0 ? Sys::_milliseconds() + timeout : timeout;

   return 0;
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
   register Variable* curLocs = m_locsStack.m_top;

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
            m_locsStack.m_top = curLocs;
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
            *sym->lvalueValue(this) = item;
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
            *m_catchBlock->target()->lvalueValue(this) = Item( ce->handler(), ce );            
            ce->decref();
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
      atomicOr(m_events, evtRaise);
   }
}


void VMContext::unhandledError( Error* ce )
{
   if( m_thrown != 0 ) m_thrown->decref();
   m_thrown = ce;

   atomicOr(m_events, evtRaise);
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

      inline virtual void describeTo( String& s, int =0 ) const { s= "#Quit"; }
   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->setTerminateEvent();
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

      inline virtual void describeTo( String& s, int =0 ) const { s= "#Complete"; }
   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->setCompleteEvent();
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

      inline virtual void describeTo( String& s, int =0 ) const { s= "#Return"; }
   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->returnFrame(Item());
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

      inline virtual void describeTo( String& s, int =0 ) const { s= "#Break"; }
   private:
      static void apply_( const PStep*, VMContext *ctx )
      {
         ctx->popCode();
         ctx->setBreakpointEvent();
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

   // do the call
   function->invoke( this, nparams );
}


void VMContext::call( Function* function, int nparams )
{
   TRACE( "Calling function %s -- codebase:%d, dynsBase:%d, stackBase:%d",
         function->locate().c_ize(),
         m_callStack.m_top->m_codeBase, 
         m_callStack.m_top->m_dynsBase, 
         m_callStack.m_top->m_stackBase );


   makeCallFrame( function, nparams );
   TRACE3( "-- codebase:%d, dynsBase:%d, stackBase:%d",
         m_callStack.m_top->m_codeBase, 
         m_callStack.m_top->m_dynsBase, 
         m_callStack.m_top->m_stackBase );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::call( Closure* closure, int32 nparams )
{
   // shall we create a full call frame?
   if( closure->closedHandler()->typeID() == FLC_CLASS_ID_FUNC ) 
   {
      Function* function = static_cast<Function*>(closure->closed());
      TRACE( "Calling closure function %s -- codebase:%d, dynsBase:%d, stackBase:%d",
         function->locate().c_ize(),
         m_callStack.m_top->m_codeBase, 
         m_callStack.m_top->m_dynsBase, 
         m_callStack.m_top->m_stackBase );

      makeCallFrame( closure, nparams );
      TRACE3( "-- codebase:%d, dynsBase:%d, stackBase:%d",
         m_callStack.m_top->m_codeBase, 
         m_callStack.m_top->m_dynsBase, 
         m_callStack.m_top->m_stackBase );

      // do the call
      function->invoke( this, nparams );
   }
   else {
      int32 locCount = (int32) closure->closedLocals();
      if( nparams < locCount ) {
         addLocals( locCount - nparams );
      }
      else if( nparams > locCount ) { 
         popData(nparams-locCount);
      }
            
      // add the closure data to the parameters.
      int totParams = locCount + closure->pushClosedData( this );
      // only functions support variable parameter call protocol;
      // -- all the others are bound to have extra params as closed data.
      closure->closedHandler()->op_call(this, totParams, closure->closed());
   }
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


void VMContext::addLocalFrame( SymbolTable* st, int pcount )
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   static Symbol* base = Engine::instance()->baseSymbol();
   
   TRACE("Add local frame PCOUNT: %d, Symbol table locals: %d, closed: %d",
      pcount, st->localCount(), st->closedCount() );
   if( st == 0 ) {
      pushCode( &stdSteps->m_localFrame );
      return;
   }
   
   // point to the item that was pushed before the parameters
   Item* top = &topData() - pcount;
   if( pcount < st->localCount() ) 
   {
      addSpace(st->localCount() - pcount);
   }

   pushCode( &stdSteps->m_localFrame );
   // 0 is marker for unused. The real base is seqId - 1.
   currentCode().m_seqId = m_dynsStack.depth()+1;
   // create the local frame in the stacks.

   // add a base marker.
   DynsData* baseDyn = m_dynsStack.addSlot();
   baseDyn->m_sym = base;
   baseDyn->m_var.value(top);
   baseDyn->m_var.base(0);
   
   // now point to the first parameter.
   ++top;
   for( int i = 0; i < st->localCount(); ++i ) 
   {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = st->getLocal(i);
      dd->m_var.value(top);
      dd->m_var.base(0);
      ++top;
   }
   
   for( int i = 0; i < st->closedCount(); ++i ) 
   {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = st->getClosed(i);
      dd->m_var.value(top);
      dd->m_var.base(0);
      ++top;
   }
}


void VMContext::unrollLocalFrame( int dynsCount )
{
   // Descend into the dynsymbol stack until we find our base.
   register DynsData* base = m_dynsStack.offset( dynsCount );
   fassert( "$base" == base->m_sym->name() );
   m_dataStack.m_top = base->m_var.value();
   m_dynsStack.m_top = base-1;
}


void VMContext::exitLocalFrame()
{
   static PStep* localFrame = &Engine::instance()->stdSteps()->m_localFrame;
   
   MESSAGE( "Exit local frame." );
   
   // Descend into the code stack until we find our local stack marker.
   register CodeFrame* base = m_codeStack.offset(currentFrame().m_codeBase);
   register CodeFrame* top = m_codeStack.m_top;
   while( top > base ) 
   {
      if( top->m_step == localFrame) 
      {
         m_codeStack.m_top = top-1;

         // if there are symbols to unroll, do it.
         // don't call unrollLocalFrame to save this call.
         if( top->m_seqId > 0 )
         {
            // the real frame is at seqId-1 as 0 is used as a marker.
            register DynsData* base = m_dynsStack.offset( top->m_seqId-1 );
            fassert( "$base" == base->m_sym->name() );
            TRACE( "Exiting with seq ID %d, data depth %d", top->m_seqId,
                 (int)(base->m_var.value() - m_dataStack.m_base) );
            m_dataStack.m_top = base->m_var.value();
            m_dynsStack.m_top = base-1;
         }
         
         break;
      }
      --top;
   }   
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

   m_locsStack.unroll( topCall->m_locsBase );
   PARANOID( "Local Symbols stack underflow at return", (m_locsStack.m_top >= m_locsStack.m_base-1) );

   // Forward the return value
   *m_dataStack.m_top = value;

   // Finalize return -- pop call frame
   // TODO: This is useful only in the interactive mode. Maybe we can use a
   // specific returnFrame for the interactive mode to achieve this.
   if( m_callStack.m_top-- ==  m_callStack.m_base )
   {
      setCompleteEvent();
      MESSAGE( "Returned from last frame -- declaring complete." );
   }

   PARANOID( "Call stack underflow at return", (m_callStack.m_top >= m_callStack.m_base-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );
}


void VMContext::forwardParams( int pcount )
{
   if( pcount > 0 )
   {
      addSpace( pcount );
      Item* base = params();
      memcpy( &topData()-pcount, base, pcount*sizeof(Item) );
   }
}


Variable* VMContext::getLValueDynSymbolVariable( const Symbol* dyns )
{
   // search for the dynsymbol in the current context.
   const CallFrame* cf = &currentFrame();
   register DynsData* dd = m_dynsStack.m_top;
   
   // keep dd from previous loop
   register DynsData* base = m_dynsStack.offset( cf->m_dynsBase );
   // on stack empty, top = base -1;
   while( dd >= base ) {
      if ( dyns->name() == dd->m_sym->name() )
      {
         // Found!
         return &dd->m_var;
      }
      
      // definitely not found?
      if( dd->m_sym->name() == "$base" ) {
         goto not_found;
      }
      --dd;      
   }
   
   // try with the locals
   fassert( cf->m_function != 0 );
   {
      Symbol* locsym = cf->m_function->symbols().findSymbol( dyns->name() );
      if( locsym != 0 )
      {
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // reference the target local variable into our slot.
         Variable* lv = locsym->getVariable(this);
         newData->m_var.set(lv->base(), lv->value());
         return &newData->m_var;
      }
   }
   
not_found:
   // not found, adding at top.
   DynsData* newData = m_dynsStack.addSlot();
   
   // save the data in the stack
   newData->m_sym = dyns;
   newData->m_var.base(0);
   newData->m_var.value(&addDataSlot());
   
   return &newData->m_var;   
}

      
Variable* VMContext::getDynSymbolVariable( const Symbol* dyns )
{
   // search for the dynsymbol in the current context.
   const CallFrame* cf = &currentFrame();
   register DynsData* dd = m_dynsStack.m_top;
   
   // no luck. Descend the frames.
   while( cf >= m_callStack.m_base )
   {
      // keep dd from previous loop
      register DynsData* base = m_dynsStack.offset( cf->m_dynsBase );
      // on stack empty, top = base -1;
      while( dd >= base ) {
         if ( dyns->name() == dd->m_sym->name() )
         {
            // Found!
            return &dd->m_var;
         }
         --dd;
      }

      // no luck, try with locals.
      fassert( cf->m_function != 0 );
      Symbol* locsym = cf->m_function->symbols().findSymbol( dyns->name() );
      if( locsym != 0 )
      {
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // We're in the same frame of the local variable.
         Variable* lv = locsym->getVariable(this);
         newData->m_var.set(lv->base(), lv->value());
         return &newData->m_var;
      }
      
      // no luck? Try with module globals.
      Module* master = cf->m_function->module();
      if( master != 0 )
      {
         Symbol* globsym = master->getGlobal( dyns->name() );
         if( globsym != 0 )
         {
            DynsData* newData = m_dynsStack.addSlot();
            newData->m_sym = dyns;
            
            // reference the target local variable into our slot.
            newData->m_var.makeReference(globsym->getVariable(this));
            return &newData->m_var;
         }
      }
      --cf;
   }   

   // still no luck? -- what about exporeted symbols in VM?
   // TODO: Correct interlocking.
   ModSpace* ms = process()->vm()->modSpace();
   if( ms != 0 )
   {
      Symbol* expsym = ms->findExportedSymbol( dyns->name() );
      if( expsym != 0 )
      {
         DynsData* newData = m_dynsStack.addSlot();
         newData->m_sym = dyns;
         // reference the target local variable into our slot.
         // we're not the context for the exported symbols
         newData->m_var.makeReference(expsym->getVariable(0));
         return &newData->m_var;
      }
   }

   // No luck at all.
   DynsData* newData = m_dynsStack.addSlot();
   
   // save the data in the stack
   newData->m_sym = dyns;
   newData->m_var.base(0);
   newData->m_var.value(&addDataSlot());
   
   return &newData->m_var;
}


Variable* VMContext::findLocalVariable( const String& name ) const
{
   // navigate through the parent symbol tables till finding the desired symbols.
   const CallFrame* cf = m_callStack.m_top;
   DynsData* dynsd = m_dynsStack.m_top;
   const CallFrame* base = m_callStack.m_base;
   
   while( cf >= base )
   {      
      fassert( cf->m_function != 0 )
      Symbol* tgtsym = cf->m_function->symbols().findSymbol( name );

      if( tgtsym != 0 )
      {
         return &m_locsStack.m_base[cf->m_locsBase + tgtsym->localId()];         
      }
      
      // no private symbol? Try with dynsyms.
      // Notice that we could do it once and for all once we fail the privates
      // but it's more likely to find the dynsym near where we're searching it,
      // and there's no extra cost in doing the search a bit at a time.
      register const DynsData* locbase = m_dynsStack.m_base + cf->m_dynsBase;      
      while( dynsd >= locbase )
      {
         if( dynsd->m_sym->name() == name ) {
            return &dynsd->m_var;
         }
         --dynsd;
      }
      
      // better luck with next time.
      --cf;
   }
   
   // not found
   return 0;
}

void VMContext::terminated()
{
   if( m_inGroup != 0 ) {
      m_inGroup->onContextTerminated(this);
   }
   else {
      // we're the main context.
      m_process->completed();
   }
}

}

/* end of vmcontext.cpp */
