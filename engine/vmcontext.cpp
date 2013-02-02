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
   memset(m_base, 0, INITIAL_STACK_ALLOC * sizeof(datatype__));
   m_top = m_base + base;
   m_max = m_base + INITIAL_STACK_ALLOC;
}

template<class datatype__>
void VMContext::LinearStack<datatype__>::init( int base, uint32 allocSize )
{
   m_base = (datatype__*) malloc( allocSize * sizeof(datatype__) );
   memset(m_base, 0, allocSize * sizeof(datatype__));
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
   m_lastRaised(0),
   m_ruleEntryResult(false),
   m_catchBlock(0),
   m_id(0),
   m_next_schedule(0),
   m_inspectible(true),
   m_bInspectMark(false),
   m_bSleeping(false),
   m_events(0),
   m_inGroup(grp),
   m_process(prc)
{
   // prepare a low-limit VM terminator request.
   m_dynsStack.init();
   m_codeStack.init();
   m_callStack.init();
   m_dataStack.init(0, m_dataStack.INITIAL_STACK_ALLOC);
   m_finallyStack.init();
   m_waiting.init();

   // create a finally base that will be never used (for performance)
   FinallyData& dt = *m_finallyStack.addSlot();
   dt.m_depth = 0;
   dt.m_finstep = 0;
   m_currentMark = 0;

   m_acquired = 0;
   pushReturn();
   m_id = prc->getNextContextID();

   Engine::collector()->registerContext(this);
}


VMContext::~VMContext()
{
   // just in case we were killed while in wait.
   abortWaits();
   acquire(0);

   if( m_lastRaised != 0 ) m_lastRaised->decref();
}


void VMContext::reset()
{
   if( m_lastRaised != 0 ) m_lastRaised->decref();
   m_lastRaised = 0;

   atomicSet(m_events, 0);

   // do not reset ingroup.

   m_catchBlock = 0;
   m_ruleEntryResult = false;

   m_dynsStack.reset();
   m_codeStack.reset();
   m_callStack.reset();
   m_dataStack.reset(0);

   abortWaits();
   acquire(0);
   m_finallyStack.init();
   m_waiting.init();

   // create a finally base that will be never used (for performance)
   FinallyData& dt = *m_finallyStack.addSlot();
   dt.m_depth = 0;
   dt.m_finstep = 0;

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
   MESSAGE( "VMContext::nextStep" );
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


void VMContext::abortWaits()
{
   Shared** base,** top;

   base = m_waiting.m_base;
   top = m_waiting.m_top+1;
   while( base != top ) {
      Shared* shared = *base;
      shared->dropWaiting( this );
      shared->decref();
      ++base;
   }
   m_waiting.m_top = m_waiting.m_base-1;
}


void VMContext::initWait()
{
   m_next_schedule = -1;
   m_waiting.m_top = m_waiting.m_base-1;
}

void VMContext::addWait( Shared* resource )
{
   *m_waiting.addSlot() = resource;
   resource->incref();
}


void VMContext::acquire(Shared* shared)
{
   if( m_acquired !=0 ) {
      m_acquired->signal();
      m_acquired->decref();
   }

   m_acquired = shared;
   if( shared != 0 ) {
      shared->incref();
   }
}


Shared* VMContext::engageWait( int64 timeout )
{
   Shared** base = m_waiting.m_base;
   Shared** top = m_waiting.m_top+1;

   TRACE( "VMContext::engageWait waiting for %d(%p) on %d shared resources in %dms.",
            id(), this,  top-base, (int) timeout  );

   while( base != top )
   {
      Shared* shared = *base;
      if (shared->consumeSignal())
      {
         abortWaits();
         TRACE( "VMContext::engageWait for %d(%p) got signaled %p%s",
                  id(), this, *base,
                     (shared->hasAcquireSemantic() ? " (with acquire semantic)" : "") );
         if(shared->hasAcquireSemantic())
         {
            acquire(shared);
         }
         return shared;
      }
      ++base;
   }

   // nothing free right now, put us at sleep.
   TRACE( "VMContext::engageWait for %d(%p) nothing signaled, will go wait", id(), this);
   setSwapEvent();

   // tell the shared we're waiting for them.
   base = m_waiting.m_base;
   top = m_waiting.m_top+1;
   while( base != top )
   {
      Shared* shared = *base;
      shared->addWaiter( this );
      ++base;
   }

   // when we want to abort wait?
   m_next_schedule = timeout > 0 ? Sys::_milliseconds() + timeout : timeout;

   return 0;
}


void VMContext::sleep( int64 timeout ) {
   TRACE( "VMContext::sleep sleeping for %d milliseconds", (int) timeout );
   m_next_schedule = timeout > 0 ? Sys::_milliseconds() + timeout : 0;
   setSwapEvent();
}


Shared* VMContext::checkAcquiredWait()
{
   Shared** base = m_waiting.m_base;
   Shared** top = m_waiting.m_top+1;

   TRACE( "VMContext::checkAcquiredWait checking %d shared resources.", top-base );

   while( base != top )
   {
      Shared* shared = *base;
      if (shared->consumeSignal())
      {
         abortWaits();
         TRACE( "VMContext::engageWait got signaled %p%s", *base,
                     (shared->hasAcquireSemantic() ? " (with acquire semantic)" : "") );
         if(shared->hasAcquireSemantic())
         {
            acquire(shared);
         }
         return shared;
      }
      ++base;
   }

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
VMContext::t_unrollResult VMContext::unrollToNext( const _checker& check )
{
   // first, we must have at least a function around.
   CallFrame* curFrame =  m_callStack.m_top;
   CodeFrame* curCode = m_codeStack.m_top;
   Item* curData = m_dataStack.m_top;
   DynsData* curDyns = m_dynsStack.m_top;

   while( m_callStack.m_base <= curFrame )
   {
      // then, get the current topCall pointer to code stack.
      CodeFrame* baseCode = m_codeStack.m_base + curFrame->m_codeBase;
      // now unroll up to when we're able to hit a next base
      while( curCode >= baseCode )
      {
         if( check( *curCode->m_step, this ) )
         {
            m_codeStack.m_top = curCode;
            m_dataStack.m_top = curData;
            m_dynsStack.m_top = curDyns;
            m_callStack.m_top = curFrame;

            // did we cross one (or more) finally handlers
            if ( static_cast<uint32>( curCode - m_codeStack.m_base ) < m_finallyStack.m_top->m_depth )
            {
               const TreeStep* finallyHandler = m_finallyStack.m_top->m_finstep;
               m_finallyStack.pop();
               check.handleFinally( this, finallyHandler );
               return e_unroll_suspended;
            }

            return e_unroll_found;
         }

         --curCode;
      }

      // are we allowed to find it just in one frame?
      if( check.dontCrossFrame() )
      {
         return e_unroll_not_found;
      }

      // did we cross one (or more) finally handlers
      if ( static_cast<uint32>( curCode - m_codeStack.m_base ) < m_finallyStack.m_top->m_depth )
      {
         m_codeStack.m_top = curCode;
         m_dataStack.m_top = curData;
         m_dynsStack.m_top = curDyns;
         m_callStack.m_top = curFrame;

         const TreeStep* finallyHandler = m_finallyStack.m_top->m_finstep;
         m_finallyStack.pop();
         check.handleFinally( this, finallyHandler );
         return e_unroll_suspended;
      }

      // unroll the call.
      curData = m_dataStack.m_base + curFrame->m_initBase;
      curDyns = m_dynsStack.m_base + curFrame->m_dynsBase;
      curFrame--;
   }

   return e_unroll_not_found;
}


class CheckIfCodeIsNextBase
{
public:
   inline bool operator()( const PStep& ps, VMContext* ) const
   {
      return ps.isNextBase();
   }

   inline bool dontCrossFrame() const { return true; }

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_unrollToNext;
      ctx->pushCode( ps );
      ctx->pushCode( handler );
   }
};


class CheckIfCodeIsLoopBase
{
public:
   inline bool operator()( const PStep& ps, VMContext* ) const
   {
      return ps.isLoopBase();
   }

   inline bool dontCrossFrame() const { return true; }

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_unrollToLoop;
      ctx->pushCode( ps );
      ctx->pushCode( handler );
   }
};



class CheckIfCodeIsCatchItem
{
public:
   CheckIfCodeIsCatchItem( const Item& item ):
      m_item(item)
   {}

   inline bool operator()( const PStep& ps, VMContext* ctx ) const
   {
      if( ps.isCatch() )
      {
         const StmtTry* stry = static_cast<const StmtTry*>( &ps );
         SynTree* st = stry->catchSelect().findBlockForItem( m_item );
         ctx->setCatchBlock( st );
         return st != 0;
      }

      return false;
   }

   inline bool dontCrossFrame() const { return false; }

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_raiseTop;

      ctx->pushData( m_item );
      ctx->pushCode( ps );
      ctx->pushCode( handler );
   }

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

   inline bool dontCrossFrame() const { return false; }

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_raiseTop;

      ctx->pushData( Item(ctx->thrownError()->handler(), ctx->thrownError()) );
      ctx->pushCode( ps );
      ctx->pushCode( handler );

   }

private:
   Class* m_err;
};


void VMContext::unrollToNextBase()
{
   CheckIfCodeIsNextBase checker;
   if( unrollToNext( checker ) == e_unroll_not_found )
   {
      raiseError( new CodeError( ErrorParam(e_continue_out, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)) );
   }
}


void VMContext::unrollToLoopBase()
{
   CheckIfCodeIsLoopBase checker;
   if( unrollToNext( checker ) == e_unroll_not_found )
   {
      raiseError( new CodeError( ErrorParam(e_break_out, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)) );
   }
}

//===================================================================
// Try frame management.
//

void VMContext::raiseItem( const Item& item )
{
   // first, if this is a boxed error, unbox it and send it to raiseError.
   if( item.isUser() )
   {
      Class* cls;
      void* inst;
      if ( item.asClassInst(cls,inst)
               && cls->isErrorClass() )
      {
         raiseError( static_cast<Error*>( inst ) );
         return;
      }
   }

   // ok, it's a real item. Just in case, remove the rising-error marker.
   if( m_lastRaised != 0 ) m_lastRaised->decref();
   m_lastRaised = 0;

   // can we catch it?
   CheckIfCodeIsCatchItem check(item);

   m_raised = item;
   m_catchBlock = 0;

   VMContext::t_unrollResult result = unrollToNext<CheckIfCodeIsCatchItem>( check );
   if( result  == e_unroll_found )
   {
      // the unroller has prepared the code for us
      fassert( m_catchBlock != 0 );
      resetCode( m_catchBlock );
      Symbol* sym = m_catchBlock->target();
      if( sym != 0 )
      {
         *resolveSymbol(sym, true) = m_raised;
      }

      //TODO: In case a finally punches in, manage it now.

   }
   else if( result == e_unroll_not_found )
   {
      // reset the raised object, anyhow.
      m_raised.setNil();

      CodeError* ce = new CodeError( ErrorParam( e_uncaught, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm));
      ce->raised( item );
      raiseError( ce );
   }
   else {
      m_raised.setNil();
   }
}

void VMContext::raiseError( Error* ce )
{
   if( m_lastRaised != 0 ) m_lastRaised->decref();
   m_lastRaised = ce;
   ce->incref();
   // can we catch it?
   m_catchBlock = 0;
   CheckIfCodeIsCatchError check( ce->handler() );

   VMContext::t_unrollResult result = unrollToNext<CheckIfCodeIsCatchError>( check );
   if( result == e_unroll_found )
   {
      resetCode( m_catchBlock );

      // assign the error to the required item.
      if( m_catchBlock->target() != 0 )
      {
         *resolveSymbol(m_catchBlock->target(), true ) = Item( ce->handler(), ce );
      }
   }
   else if( result == e_unroll_not_found )
   {
      // we're out of business; ask to raise to our parent.
      atomicOr( m_events, evtRaise );
   }
   // otherwise, the throw is suspended
   else {
      atomicAnd( m_events, ~evtRaise );
      ce->decref();
      m_lastRaised = 0;
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


void VMContext::callInternal( Function* function, int nparams, const Item& self )
{
   TRACE( "Calling method %s.%s -- call frame code:%p, data:%p, call:%p",
         self.describe(3).c_ize(), function->locate().c_ize(),
         m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   makeCallFrame( function, nparams, self );

   // do the call
   function->invoke( this, nparams );
}


void VMContext::callInternal( Function* function, int nparams )
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


void VMContext::callInternal( Closure* closure, int32 nparams )
{
   // shall we create a full call frame?
   Function* function = closure->closed();
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


void VMContext::callItem( const Item& item, int pcount, Item const* params )
{
   TRACE( "Calling item: %s -- call frame code:%p, data:%p, call:%p",
      item.describe(2).c_ize(), m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   pushData( item );
   Class* cls;
   void* data;
   item.forceClassInst( cls, data );

   if( pcount > 0 )
   {
      addSpace( pcount);
      memcpy( m_dataStack.m_top-pcount, params, pcount * sizeof(item) );
   }

   cls->op_call( this, pcount, data );
}

void VMContext::call( Function* func, int pcount, Item const* params )
{
   TRACE( "VMContext::call function: %s -- call frame code:%p, data:%p, call:%p",
     func->name().c_ize(), m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   pushData( Item( func->handler(), func) );

   if( pcount > 0 )
   {
      addSpace(pcount);
      memcpy( m_dataStack.m_top-pcount, params, pcount * sizeof(Item) );
   }
   makeCallFrame(func, pcount);
   func->invoke(this, pcount);
}


void VMContext::call( Function* func, const Item& self, int pcount, Item const* params )
{
   TRACE( "VMContext::call method: %s.%s -- call frame code:%p, data:%p, call:%p",
     self.describe().c_ize(), func->name().c_ize(), m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   pushData( Item( func->handler(), func) );

   if( pcount > 0 )
   {
      addSpace(pcount);
      memcpy( m_dataStack.m_top-pcount, params, pcount * sizeof(Item) );
   }
   makeCallFrame(func, pcount, self);
   func->invoke(this, pcount);
}

void VMContext::call( Closure* cls, int pcount, Item const* params )
{
   TRACE( "VMContext::call closure: %s -- call frame code:%p, data:%p, call:%p",
           cls->closed()->name().c_ize(), m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   pushData( Item( cls->closed()->handler(), cls) );

   if( pcount > 0 )
   {
      addSpace(pcount);
      memcpy( m_dataStack.m_top-pcount, params, pcount * sizeof(Item) );
   }
   makeCallFrame(cls, pcount );
   cls->closed()->invoke(this, pcount);
}


void VMContext::addLocalFrame( VarMap* st, int pcount )
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   static Symbol* base = Engine::instance()->baseSymbol();
   
   if( st != 0 ) {
      TRACE("Add local frame PCOUNT: %d/%d, Symbol table locals: %d, closed: %d",
               pcount, st->paramCount(), st->localCount(), st->closedCount() );
   }
   else {
      TRACE("Add local frame PCOUNT: %d (no table)", pcount );
   }

   pushCode( &stdSteps->m_localFrame );
   // 0 is marker for unused. The real base is seqId - 1.
   currentCode().m_seqId = m_dynsStack.depth()+1;
   // create the local frame in the stacks.

   // add a base marker.
   DynsData* baseDyn = m_dynsStack.addSlot();
   baseDyn->m_sym = base;
   baseDyn->m_value = m_dataStack.m_top;
   
   // if we don't have a map, there's nothing else we should do.
   if( st == 0 ) {
      popData( pcount + 1 );
      return;
   }

   // Assign the parameters
   Item* top = &topData() - pcount+1;
   int32 p = 0;
   while( p < pcount )
   {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = Engine::getSymbol(st->getParamName(p), false);
      dd->m_value = top;
      ++top;
      ++p;
   }

   // blank unused parameters
   pcount = st->paramCount();
   while( p < pcount ) {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = Engine::getSymbol(st->getParamName(p), false);
      dd->m_value = &dd->m_internal;
      dd->m_internal.setNil();
      ++p;
   }
}


void VMContext::unrollLocalFrame( int dynsCount )
{
   // Descend into the dynsymbol stack until we find our base.
   register DynsData* base = m_dynsStack.offset( dynsCount );
   fassert( "$base" == base->m_sym->name() );
   m_dataStack.m_top = base->m_value;
   m_dynsStack.m_top = base-1;
}


void VMContext::exitLocalFrame( bool exec )
{
   static PStep* localFrame = &Engine::instance()->stdSteps()->m_localFrame;
   static PStep* localFrameExec = &Engine::instance()->stdSteps()->m_localFrameExec;
   
   MESSAGE( "Exit local frame." );
   
   // Descend into the code stack until we find our local stack marker.
   register CodeFrame* base = m_codeStack.offset(currentFrame().m_codeBase);
   register CodeFrame* top = m_codeStack.m_top;
   while( top > base ) 
   {
      if( top->m_step == localFrame) 
      {
         // we'll be here again...
         if( static_cast<uint32>(top - m_codeStack.m_base) < m_finallyStack.m_top->m_depth )
         {
            const TreeStep* fd = m_finallyStack.m_top->m_finstep;
            m_finallyStack.pop();
            m_codeStack.m_top = top;
            if( exec ) {
               // chance into the exec frame, keep depth in m_seq
               m_codeStack.m_top->m_step = localFrameExec;
            }
            pushCode(fd);
            return;
         }

         m_codeStack.m_top = top-1;
         // if there are symbols to unroll, do it.
         if( top->m_seqId > 0 )
         {
            Item td = topData();
            unrollLocalFrame( top->m_seqId-1 );
            topData() = td;
         }

         break;
      }
      --top;
   }   
}


void VMContext::removeData( uint32 pos, uint32 removeSize )
{
   Item* base = m_dataStack.m_top-pos;

   memmove( base,
            base + removeSize,
            sizeof(Item) * (pos-removeSize+1) );

   m_dataStack.m_top -= removeSize;
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

template<class _returner>
void VMContext::returnFrame_base( const Item& value )
{
   static PStep* ps_return = _returner::pstep();

   register CallFrame* topCall = m_callStack.m_top;
   TRACE1( "Return frame from function %s", topCall->m_function->name().c_ize() );

   if( topCall->m_codeBase < m_finallyStack.m_top->m_depth )
   {
      const TreeStep* fd = m_finallyStack.m_top->m_finstep;
      m_finallyStack.pop();
      pushData(value);
      pushCode(ps_return);
      pushCode(fd);
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
      setCompleteEvent();
      MESSAGE( "Returned from last frame -- declaring complete." );
   }

   PARANOID( "Call stack underflow at return", (m_callStack.m_top >= m_callStack.m_base-1) );
   TRACE( "Return frame code:%p, data:%p, call:%p", m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   _returner::post_return( this );
}

class ReturnerSimple
{
public:
   inline static PStep* pstep() {
      return &Engine::instance()->stdSteps()->m_returnFrameWithTop;
   }

   inline static void post_return( VMContext* ) {
   }
};

class ReturnerND
{
public:
   inline static PStep* pstep() {
      return &Engine::instance()->stdSteps()->m_returnFrameWithTopDoubt;
   }

   inline static void post_return( VMContext* ctx ) {
      ctx->SetNDContext();
   }
};

class ReturnerEval
{
public:
   inline static PStep* pstep() {
      return &Engine::instance()->stdSteps()->m_returnFrameWithTopEval;
   }

   inline static void post_return( VMContext* ctx ) {
     Class* cls = 0;
      void* data = 0;
      ctx->topData().forceClassInst(cls, data);
      cls->op_call( ctx, 0, data );
   }
};

class ReturnerNDEval
{
public:
   inline static PStep* pstep() {
      return &Engine::instance()->stdSteps()->m_returnFrameWithTopEval;
   }

   inline static void post_return( VMContext* ctx ) {
      ctx->SetNDContext();
      Class* cls = 0;
      void* data = 0;
      ctx->topData().forceClassInst(cls, data);
      cls->op_call( ctx, 0, data );
   }
};


void VMContext::returnFrame( const Item& value )
{
   returnFrame_base<ReturnerSimple>(value);
}

void VMContext::returnFrameND( const Item& value )
{
   returnFrame_base<ReturnerND>(value);
}

void VMContext::returnFrameEval( const Item& value )
{
   returnFrame_base<ReturnerEval>(value);
}

void VMContext::returnFrameNDEval( const Item& value )
{
   returnFrame_base<ReturnerEval>(value);
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


void VMContext::defineSymbol( Symbol* sym, Item* data )
{
   DynsData* newData = m_dynsStack.addSlot();
   newData->m_sym = sym;
   newData->m_value = data;
}


void VMContext::defineSymbol( Symbol* sym)
{
   DynsData* newData = m_dynsStack.addSlot();
   newData->m_sym = sym;
   newData->m_value = &newData->m_internal ;
}


Item* VMContext::resolveSymbol( const Symbol* dyns, bool forAssign )
{
   TRACE1( "VMContext::resolveSymbol -- resolving %s symbol \"%s\"%s",
            (dyns->isGlobal() ? "global": "local"),
            dyns->name().c_ize(),
            (forAssign ? " (for assign)": " (for access)") );

   static Symbol* baseSym = Engine::instance()->baseSymbol();

   // search for the dynsymbol in the current context.
   const CallFrame* cf = &currentFrame();
   register DynsData* dd = m_dynsStack.m_top;

   // search resolved symbols in current frame.
   // when is forassign, we must stop at topmost symbol base.
   register DynsData* dbase = m_dynsStack.m_base + cf->m_dynsBase;
   while( dd >= dbase )
   {
      // found?
      if ( dyns == dd->m_sym ) {
         TRACE2( "VMContext::resolveSymbol -- \"%s\" already resolved as %p", dyns->name().c_ize(), dd->m_value );
         return dd->m_value;
      }
      // arrived at a local base?
      if ( baseSym == dd->m_sym )
      {
         // in the end, we must create a new assignable slot.
         // notice that evaluation parameters are above the base symbol.
         DynsData* newSlot = m_dynsStack.addSlot();
         newSlot->m_sym = dyns;
         newSlot->m_internal.setNil();
         newSlot->m_value = &newSlot->m_internal;
         TRACE2( "VMContext::resolveSymbol -- \"%s\" went down to a local base, creating new.", dyns->name().c_ize() );
         return newSlot->m_value;
      }

      --dd;
   }

   // if we're here, we didn't find it -- it might be an unresolved local variable...
   const String& name = dyns->name();
   Variable* lvar = cf->m_function->variables().find(name);
   if( lvar != 0 )
   {
      Item* resolved = 0;
      switch( lvar->type() ) {
      case Variable::e_nt_closed:
         if( cf->m_closure != 0 )
         {
            resolved = cf->m_closure->get(name);
         }
         break;

      case Variable::e_nt_local:
         if( forAssign )
         {
            resolved = local(lvar->id());
         }
         break;

      case Variable::e_nt_param:
         resolved = param(lvar->id());
         break;

      default:
         fassert2( false, "Shouldn't have this in a function" );
         break;
      }

      if( resolved != 0 )
      {
         DynsData* newSlot = m_dynsStack.addSlot();
         newSlot->m_sym = dyns;
         newSlot->m_internal.setNil();
         newSlot->m_value = resolved;
         TRACE2( "VMContext::resolveSymbol -- \"%s\" Found in locals as %p (%s)",
                  dyns->name().c_ize(), resolved, resolved->describe().c_ize() );
         return resolved;
      }
   }

   // No luck at all; try to resolve the variable in the global arena
   Item* var = resolveVariable( dyns->name(), true, forAssign );

   DynsData* newSlot;

   if( var == 0 )
   {
      if( ! forAssign ) {
         throw new CodeError( ErrorParam(e_undef_sym,
                  currentCode().m_step->sr().line(),
                  cf->m_function->module() ? cf->m_function->module()->name() : "" )
                  .symbol( cf->m_function->name() )
                  .extra(dyns->name())
                  );
      }

      // not found? -- it's an unbound symbol.
      newSlot = m_dynsStack.addSlot();
      var = &newSlot->m_internal;
      var->setNil();
      TRACE2( "VMContext::resolveSymbol -- \"%s\" Not found, creating new.",
               dyns->name().c_ize() );
   }
   else {
      newSlot = m_dynsStack.addSlot();
   }
   
   newSlot->m_sym = dyns;
   newSlot->m_value = var;

   TRACE2( "VMContext::resolveSymbol -- \"%s\" Resolved as new/global/extern %p (%s)",
            dyns->name().c_ize(), var, var->describe().c_ize() );

   return var;
}


Item* VMContext::resolveVariable( const String& name, bool isGlobal, bool forAssign )
{
   TRACE1( "VMContext::resolveVariable -- resolving %s%s", name.c_ize(), isGlobal ? " (global)": "")
   CallFrame& cf = currentFrame();
   Function* func = cf.m_function;

   if( ! isGlobal )
   {
      Variable* var = func->variables().find( name );
      if( var != 0 ) {
         switch( var->type() ) {
         case Variable::e_nt_closed:
            if( cf.m_closure == 0 ) return 0;
            return cf.m_closure->get(name);

         case Variable::e_nt_local:
            return local(var->id());

         case Variable::e_nt_param:
            return param(var->id());

         default:
            fassert2( false, "Shouldn't have this in a function" );
            break;
         }
      }
      else if( forAssign ) {
         return 0;
      }
    }

   // didn't find it locally, try globally
   Module* mod = func->module();
   if( mod != 0 ) {
      // findGlobal will find also externally resolved variables.
      Item* global = mod->getGlobalValue( name );
      if( global != 0 ) {
         return global;
      }
      else if( forAssign ) {
         Variable* var = mod->addGlobal( name, Item(), false );
         global = mod->getGlobalValue( var->id() );
         return global;
      }
   }

   // if the function is an eta...
   if( func->isEta() && callDepth() > 0 )
   {
      // we must inspect also the global context of the caller.
      Function* caller = callerFrame(1).m_function;
      if ( caller->module() != 0 )
      {
         Item* global = caller->module()->getGlobalValue( name );
         if( global != 0 ) {
            return global;
         }
      }
   }

   // try as non-imported extern
   if( ! forAssign )
   {
      // if the module space is the same as the vm modspace,
      // mod->findGlobal has already searched for it
      Item* item = vm()->modSpace()->findExportedValue( name );
      if( item != 0 ) {
         return item;
      }
   }

   // no luck
   return 0;
}


ClosedData* VMContext::getTopClosedData() const
{
   CallFrame *cf = m_callStack.m_top;
   while( cf >= m_callStack.m_base ) {
      if( cf->m_closingData != 0 ) {
         return cf->m_closingData;
      }
      --cf;
   }

   return 0;
}


Item* VMContext::findLocal( const String& name ) const
{
   CallFrame *cf = m_callStack.m_top;
   while( cf >= m_callStack.m_base ) {

      Variable* var = cf->m_function->variables().find(name);
      if( var != 0 ) {
         switch( var->type() ) {
         case Variable::e_nt_local:
            return &m_dataStack.m_base[ var->id() + cf->m_stackBase + cf->m_paramCount ];

         case Variable::e_nt_param:
            return &m_dataStack.m_base[ var->id() + cf->m_stackBase ];

         case Variable::e_nt_closed:
            if( cf->m_closure != 0 ) {
               return cf->m_closure->get(name);
            }
            break;

         default:
            break;
         }
      }

      --cf;
   }

   return 0;
}


void VMContext::terminated()
{
   //... and from the manager
   vm()->contextManager().onContextTerminated(this);

   int value;
   value = atomicFetch( m_events );
   if( (value & evtRaise) )
   {
      if( m_inGroup != 0 ) {
         m_inGroup->setError(this->m_lastRaised);
         m_inGroup->onContextTerminated(this);
      }
      else {
         // we're the main context.
         m_process->completedWithError(this->m_lastRaised);
      }
   }
   else {
      if( m_inGroup != 0 ) {
         m_inGroup->onContextTerminated(this);
      }
      else {
         // we're the main context.
         m_process->setResult( topData() );
         m_process->completed();
      }
   }

   // we're off from the collector...
   Engine::collector()->unregisterContext(this);
}


Error* VMContext::runtimeError( int id, const String& extra, int line )
{
   String noname;
   Function* curFunc = currentFrame().m_function;
   const String* modName = curFunc->module() == 0 ? &noname : &curFunc->module()->name();

   if( line == 0 ) {
      line = currentCode().m_step->sr().line();
   }

   CodeError* error = new CodeError( ErrorParam(id, line, *modName )
            .origin(ErrorParam::e_orig_runtime)
            .symbol( curFunc->name() )
            .extra( extra ) );

   return error;
}

void VMContext::contestualize( Error* error )
{
   String noname;
   Function* curFunc = currentFrame().m_function;
   const String* modName = curFunc->module() == 0 ? &noname : &curFunc->module()->name();
   int line = currentCode().m_step->sr().line();

   error->line(line);
   error->module(*modName);
   error->symbol( curFunc->name() );
}

void VMContext::gcStartMark( uint32 mark )
{
   if( m_currentMark != mark )
   {
      m_currentMark = mark;
   }
}

void VMContext::gcPerformMark()
{
   uint32 mark = m_currentMark;
   // first, mark the items.
   {
      Item* base = m_dataStack.m_base;
      while( base <= m_dataStack.m_top ) {
         base->gcMark(mark);
         ++base;
      }
   }

   // then, the symbols.
   {
      DynsData* base = m_dynsStack.m_base;
      while( base <= m_dynsStack.m_top ) {
         if( base->m_value != 0 ) {
            base->m_value->gcMark(mark);
         }
         ++base;
      }
   }

   // items in call frames
   {
      CallFrame* base = m_callStack.m_base;
      while( base <= m_callStack.m_top )
      {
         base->m_self.gcMark(mark);
         base->m_function->gcMark(mark);

         if( base->m_closingData != 0 ) {
            base->m_closingData->gcMark(mark);
         }
         if( base->m_closure != 0 ) {
            base->m_closure->gcMark(mark);
         }
         ++base;
      }
   }

   // then, other various elements.
   {
      vm()->modSpace()->gcMark(mark);
   }
}

void VMContext::setInspectEvent()
{
   m_inspectible = true;  // not really necessary

   m_mtx_sleep.lock();
   m_bInspectMark = true;
   if( m_bSleeping ) {
      m_mtx_sleep.unlock();
      vm()->contextManager().wakeUp(this);
   }
   else {
      m_mtx_sleep.unlock();
   }
   atomicOr(m_events, evtSwap);
}

bool VMContext::goToSleep()
{
   m_mtx_sleep.lock();
   if( m_bInspectMark ) {
      m_mtx_sleep.unlock();
      return false;
   }
   m_bSleeping = true;
   m_mtx_sleep.unlock();
   return true;
}

void VMContext::awake()
{
   m_mtx_sleep.lock();
   m_bSleeping = false;
   m_mtx_sleep.unlock();
}


void VMContext::onStackRebased( Item* oldBase )
{
   TRACE( "VMContext::onStackRebased %p -> %p", oldBase, m_dataStack.m_base );

   // rebase the dynsym satack.
   Item* newBase = m_dataStack.m_base;
   Item* oldTop = oldBase + (m_dataStack.m_top - newBase);

   DynsData* dt = m_dynsStack.m_base;
   DynsData* endDt = m_dynsStack.m_top;

   while( dt <= endDt )
   {
      if( dt->m_value >= oldBase && dt->m_value <= oldTop ) {
         dt->m_value = newBase + (dt->m_value - oldBase);
      }
      ++dt;
   }
}


}

/* end of vmcontext.cpp */
