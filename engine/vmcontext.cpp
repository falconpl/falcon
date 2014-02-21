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
#include <falcon/gctoken.h>
#include <falcon/itemstack.h>

#include <falcon/module.h>       // For getDynSymbolValue
#include <falcon/modspace.h>

#include <falcon/storer.h>
#include <falcon/stderrors.h>
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
   m_allocSize = INITIAL_STACK_ALLOC;
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
   m_status(statusBorn),
   m_lastRaised(0),
   m_catchBlock(0),
   m_id(0),
   m_next_schedule(0),
   m_inspectible(true),
   m_bInspectMark(false),
   m_bSleeping(false),
   m_events(0),
   m_suspendedEvents(0),
   m_inGroup(grp),
   m_process(prc),
   m_caller(0)
{
   m_newTokens = new GCToken(0,0);
   m_newTokens->m_next = m_newTokens;
   m_newTokens->m_prev = m_newTokens;

   m_dynsStack.init();
   m_codeStack.init();
   m_callStack.init();
   m_dataStack.init(0, m_dataStack.INITIAL_STACK_ALLOC);
   m_finallyStack.init();
   m_waiting.init();
   m_itemStack = new ItemStack(prc->itemPagePool());

   m_id = prc->getNextContextID();
   m_acquired = 0;
   m_signaledResource = 0;

   m_firstWeakRef = 0;

   pushBaseElements();

   // ready to go
   Engine::collector()->registerContext(this);
}


VMContext::~VMContext()
{
   // just in case we were killed while in wait.
   abortWaits();
   releaseAcquired();
   clearSignaledResource();

   m_newTokens->m_prev->m_next = 0;
   GCToken* token = m_newTokens;
   while (token != 0 )
   {
      GCToken* old = token;
      token = token->m_next;
      delete old;
   }

   if( m_lastRaised != 0 ) m_lastRaised->decref();
}


void VMContext::reset()
{
   if( m_lastRaised != 0 ) m_lastRaised->decref();
   m_lastRaised = 0;
   m_caller = 0;

   atomicSet(m_events, 0);
   setStatus(statusBorn);

   // do not reset ingroup.

   abortWaits();
   releaseAcquired();
   clearSignaledResource();

   m_catchBlock = 0;
   m_inspectible = true;
   m_bInspectMark = false;
   m_bSleeping = false;

   m_dynsStack.reset();
   m_codeStack.reset();
   m_callStack.reset();
   m_dataStack.reset(0);
   m_finallyStack.reset();
   m_waiting.reset();

   pushBaseElements();
   clearEvents();

   if( m_lastRaised != 0 )
   {
      m_lastRaised->decref();
      m_lastRaised = 0;
   }

   m_raised.setNil();
}


void VMContext::registerOnTerminate( VMContext::WeakRef* subscriber )
{
   m_mtxWeakRef.lock();
   subscriber->m_next = m_firstWeakRef;
   subscriber->m_prev = 0;

   // adding before first.
   if( m_firstWeakRef == 0)
   {
      m_firstWeakRef = subscriber;
   }
   else {
      m_firstWeakRef->m_prev = subscriber;
      m_firstWeakRef = subscriber;
   }
   m_mtxWeakRef.unlock();
}


void VMContext::unregisterOnTerminate( VMContext::WeakRef* subscriber )
{
   m_mtxWeakRef.lock();
   if( subscriber == m_firstWeakRef )
   {
      m_firstWeakRef = m_firstWeakRef->m_next;
      if( m_firstWeakRef != 0 )
      {
         m_firstWeakRef->m_prev = 0;
      }
   }
   else {
      if( subscriber->m_prev != 0 )
      {
         subscriber->m_prev->m_next = subscriber->m_next;
      }

      if(subscriber->m_next != 0 )
      {
         subscriber->m_next->m_prev = subscriber->m_prev;
      }
   }
   m_mtxWeakRef.unlock();
}


void VMContext::pushBaseElements()
{
   static const PStep* endContext = &Engine::instance()->stdSteps()->m_endOfContext;

   // create a finally base that will be never used (for performance)
   FinallyData& dt = *m_finallyStack.addSlot();
   dt.m_depth = 0;
   dt.m_finstep = 0;
   m_currentMark = 0;

   // create a code that will be never used (for performance)
   // also, ensures that the VM is quitted if it hits this
   CodeFrame* cf = m_codeStack.addSlot();
   cf->m_step = endContext;
}


bool VMContext::location( LocationInfo& infos ) const
{
   // location is given by current function and its module plus current source line.
   if( codeEmpty() || callDepth() == 0 )
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

const PStep* VMContext::nextStep( int frame ) const
{
   MESSAGE( "VMContext::nextStep" );
   if( codeEmpty() )
   {
      return 0;
   }
   PARANOID( "Call stack empty", (this->callDepth() > 0) );

   const CodeFrame* cframe;

   if( frame == 0 )
   {
      cframe = m_codeStack.m_top;
   }
   else {
      CallFrame& callf = callerFrame(frame-1);
      cframe = m_codeStack.m_base + callf.m_codeBase;
   }

   const PStep* ps = cframe->m_step;
   if( ps->isComposed() )
   {
      const SynTree* st = static_cast<const SynTree*>(ps);
      return st->at(cframe->m_seqId);
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
   while( base < top ) {
      Shared* shared = *base;
      shared->dropWaiting( this );
      shared->decref();
      ++base;
   }
   m_waiting.m_top = m_waiting.m_base-1;
}

void VMContext::clearWaits()
{
   Shared** base,** top;

   base = m_waiting.m_base;
   top = m_waiting.m_top+1;
   while( base != top ) {
      Shared* shared = *base;
      shared->decref();
      ++base;
   }
   m_waiting.m_top = m_waiting.m_base-1;
}

void VMContext::initWait()
{
   m_next_schedule = 0;
   m_waiting.m_top = m_waiting.m_base-1;
}

void VMContext::addWait( Shared* resource )
{
   *m_waiting.addSlot() = resource;
   resource->incref();
}


void VMContext::acquire(Shared* shared)
{
   fassert( m_acquired == 0 );
   fassert( shared != 0 );

   m_acquired = shared;
   shared->incref();
}


bool VMContext::releaseAcquired()
{
   if( m_acquired != 0 )
   {
      m_acquired->signal();
      m_acquired->decref();
      m_acquired = 0;
      if( m_suspendedEvents != 0 )
      {
         atomicOr( m_events, m_suspendedEvents );
         m_suspendedEvents = 0;
         return true;
      }
   }

   return false;
}


void VMContext::signaledResource( Shared* shared )
{
   if( m_signaledResource != 0 )
   {
      m_signaledResource->decref();
   }

   shared->incref();
   m_signaledResource = shared;
}


Shared* VMContext::getSignaledResouce()
{
   Shared* signaled = m_signaledResource;
   m_signaledResource = 0;
   return signaled;
}


void VMContext::clearSignaledResource()
{
   if( m_signaledResource != 0 )
   {
      m_signaledResource->decref();
      m_signaledResource = 0;
   }
}


Shared* VMContext::engageWait( int64 timeout )
{
   Shared** base = m_waiting.m_base;
   Shared** top = m_waiting.m_top+1;

   // we have sereral no-op exit points, better to clear the schedule.
   m_next_schedule = 0;

   TRACE( "VMContext::engageWait waiting for %d(%p) on %ld shared resources in %dms.",
            id(), this,  (long) (top-base), (int) timeout  );

   if(base == top)
   {
      // not waiting on nothing.
      return 0;
   }

   while( base != top )
   {
      Shared* shared = *base;
      if (shared->consumeSignal(this,1) > 0)
      {
         clearWaits();
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

   // was this just a try?
   if( timeout == 0 )
   {
      clearWaits();
      return 0;
   }

   // nothing free right now, put us at sleep.
   TRACE( "VMContext::engageWait for %d(%p) nothing signaled, will go wait", id(), this);
   setSwapEvent();
   // just in case someone forgot...
   clearSignaledResource();

   // when we want to abort wait?
   int64 randesVousAt = timeout > 0 ? Sys::_milliseconds() + timeout : timeout;
   m_next_schedule = randesVousAt;

   return 0;
}


int32 VMContext::waitingSharedCount() const
{
   return (int32) ((m_waiting.m_top+1) - m_waiting.m_base);
}

Shared* VMContext::declareWaits()
{
   // tell the shared we're waiting for them.
   Shared** base = m_waiting.m_base;
   Shared** current = m_waiting.m_base;
   Shared** top = m_waiting.m_top;
   if( current > top && m_next_schedule < 0 )
   {
      m_next_schedule = 0;
      return 0;
   }

   while( current <= top )
   {
        Shared* shared = *current;
        if( shared->addWaiter( this ) )
        {
           // oh, we made it at last.
           while( --current >= base )
           {
              shared->dropWaiting(this);
           }

           clearWaits();
           return shared;
        }
        ++current;
   }

   return 0;
}

void VMContext::sleep( int64 timeout )
{
   TRACE( "VMContext::sleep sleeping for %d milliseconds", (int) timeout );
   m_next_schedule = timeout > 0 ? Sys::_milliseconds() + timeout : 0;

   // release acquired resources.
   releaseAcquired();
   setSwapEvent();
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
   TRACE1( "VMContext::startRuleFrame -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
            dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();
   DynsData* slot = m_dynsStack.addSlot();

   register Item& frame = *m_itemStack->push(m_dynsStack.depth());
   slot->m_sym = base;
   slot->m_value = &frame; // overkill
   frame.type(FLC_ITEM_FRAMING);
   int64 dataDepth = m_dataStack.depth();
   dataDepth <<= 32;
   // StmtRule or the rule master is ALREADY in the code stack,
   // so we have to record the previous depth m_codeStack.depth()-1
   uint32 codeDepth = m_codeStack.depth()-1;
   frame.content.data.val64 = dataDepth | codeDepth;
   frame.flags(1);

}


void VMContext::startRuleNDFrame( uint32 tbPoint )
{
   TRACE1( "VMContext::startRuleNDFrame -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
            dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();
   DynsData* slot = m_dynsStack.addSlot();

   register Item& frame = *m_itemStack->push(m_dynsStack.depth());
   slot->m_sym = base;
   slot->m_value = &frame; // overkill
   frame.type(FLC_ITEM_FRAMING);
   int64 dataDepth = m_dataStack.depth();
   dataDepth <<= 32;
   frame.content.data.val64 = dataDepth | tbPoint;
   frame.flags(0);
}

uint32 VMContext::unrollRuleNDFrame()
{
   TRACE2( "VMContext::unrollRuleNDFrame -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
            dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();

   DynsData* dbase = m_dynsStack.m_base + currentFrame().m_dynsBase;
   DynsData* top = m_dynsStack.m_top;
   while( top >= dbase )
   {
      if( top->m_sym == base )
      {
         register Item& frame = *top->m_value;
         int64 depth = frame.content.data.val64;
         m_dataStack.unroll( 0xFFFFFFFF & (depth >> 32));

         // return 0xFFFFFFFF or the real depth depending on the flags.
         uint32 ret;
         if( frame.flags() == 0 )
         {
            // try again
            ret = (depth & 0xFFFFFFFF);
            m_dynsStack.m_top = top-1;
         }
         else {
            // nowhere else to go.
            ret = 0xFFFFFFFF;
            m_dynsStack.m_top = top;
         }

         TRACE1( "VMContext::unrollRuleNDFrame -- after unroll dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
                     dataSize(), m_dynsStack.depth(), m_codeStack.depth());
         return ret;
      }

      top--;
   }

   return 0xFFFFFFFF;
}


void VMContext::dropRuleNDFrames()
{
   TRACE2( "VMContext::dropRuleNDFrames -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
            dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();
   static const Symbol* baseIgnore = Engine::instance()->ruleBaseSymbol();

   DynsData* dbase = m_dynsStack.m_base + currentFrame().m_dynsBase;
   DynsData* top = m_dynsStack.m_top;
   while( top >= dbase )
   {
      if( top->m_sym == base )
      {
         register Item& frame = *top->m_value;

         if( frame.flags() == 1 )
         {
            break;
         }
         else {
            top->m_sym = baseIgnore;
         }
      }

      top--;
   }
}


void VMContext::unrollRule()
{
   TRACE1( "VMContext::unrollRule -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
                  dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();

   DynsData* dbase = m_dynsStack.m_base + currentFrame().m_dynsBase;
   DynsData* top = m_dynsStack.m_top;
   while( top >= dbase )
   {
      if( top->m_sym == base )
      {
         register Item& frame = *top->m_value;
         // we're interested in the ruleframe only
         if( frame.flags() == 1 )
         {
            int64 depth = frame.content.data.val64;
            m_dataStack.unroll( 0xFFFFFFFF & (depth >> 32));
            m_dynsStack.m_top = top - 1;
            // in this case, we're iterested in removing the rule frame too.
            m_codeStack.unroll( depth & 0xFFFFFFFF );

            TRACE1( "VMContext::unrollRule -- after unroll dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
                        dataSize(), m_dynsStack.depth(), m_codeStack.depth());
            return;
         }
      }

      top--;
   }

   fassert2( false, "Base rule frame not found" );
}


void VMContext::commitRule()
{
   TRACE1( "VMContext::commitRule -- dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
               dataSize(), m_dynsStack.depth(), m_codeStack.depth());

   static const Symbol* base = Engine::instance()->baseSymbol();

   DynsData* dbase = m_dynsStack.m_base + currentFrame().m_dynsBase;
   DynsData* top = m_dynsStack.m_top;
   while( top >= dbase )
   {
      if( top->m_sym == base )
      {
         register Item& frame = *top->m_value;
         if( frame.flags() == 1 )
         {
            int64 depth = frame.content.data.val64;
            m_dataStack.unroll( 0xFFFFFFFF & (depth >> 32));
            m_codeStack.unroll( depth & 0xFFFFFFFF );
            m_dynsStack.m_top = top - 1;
            TRACE1( "VMContext::commitRule -- after unroll dataDepth: %ld; dynsDepth: %ld; codeDepth: %ld",
                        dataSize(), m_dynsStack.depth(), m_codeStack.depth());
            return;
         }
      }
      else
      {
         // apply the variable down, copying its values to previous frames.
         register const Symbol* sym = top->m_sym;
         DynsData* top1 = top-1;
         while( top1 >= dbase )
         {
            if( top1->m_sym == sym )
            {
               *top1->m_value = *top->m_value;
               // if the're are other copies of the symbol downstairs,
               // we'll catch them later.
               break;
            }

            --top1;
         }
      }

      top--;
   }

   fassert2( false, "Base rule frame not found" );
}



template<class _checker>
VMContext::t_unrollResult VMContext::unrollToNext( const _checker& check )
{
   // first, we must have at least a function around.
   CallFrame* curFrame =  m_callStack.m_top;
   CodeFrame* curCode = m_codeStack.m_top;

   while( m_callStack.m_base <= curFrame )
   {
      // then, get the current topCall pointer to code stack.
      CodeFrame* baseCode = m_codeStack.m_base + curFrame->m_codeBase;
      // now unroll up to when we're able to hit a next base
      while( curCode >= baseCode )
      {
         if( check( *curCode->m_step, this ) )
         {
            fassert2(curCode->m_dataDepth != 0xFFFFFFFF, "Data unroll uninitialized" );
            fassert2(curCode->m_dynsDepth != 0xFFFFFFFF, "Dynsstack unroll uninitialized" );
            m_codeStack.m_top = curCode;
            m_dataStack.unroll(curCode->m_dataDepth);
            m_dynsStack.unroll(curCode->m_dynsDepth);
            m_callStack.m_top = curFrame;

            // did we cross one (or more) finally handlers
            uint32 depth = static_cast<uint32>( curCode - m_codeStack.m_base );
            if ( depth < m_finallyStack.m_top->m_depth )
            {
               const TreeStep* finallyHandler = m_finallyStack.m_top->m_finstep;
               m_finallyStack.pop();
               check.handleFinally( this, finallyHandler );
               return e_unroll_suspended;
            }

            // perform required cleanups after unroll (if any)
            check.onBaseFound( this );

            // report success
            return e_unroll_found;
         }

         --curCode;
      }

      // are we allowed to find it just in one frame?
      if( check.dontCrossFrame() )
      {
         return e_unroll_not_found;
      }

      // did we cross one (or more) finally handlers in the current call frame?
      if ( static_cast<uint32>( curCode - m_codeStack.m_base ) < m_finallyStack.m_top->m_depth )
      {
         check.onCrossFinally(this);
         // set the call frame.
         m_callStack.m_top = curFrame;

         // set the code stack accordingly to the finally step
         const TreeStep* finallyHandler = m_finallyStack.m_top->m_finstep;
         m_codeStack.unroll( m_finallyStack.m_top->m_depth );
         CodeFrame* curCode = m_codeStack.m_top;

         fassert2(curCode->m_dataDepth != 0xFFFFFFFF, "Data unroll uninitialized" );
         fassert2(curCode->m_dynsDepth != 0xFFFFFFFF, "Dynsstack unroll uninitialized" );

         // unroll data and dyns frame accordingly
         m_dataStack.unroll( curCode->m_dataDepth );
         m_dynsStack.unroll( curCode->m_dynsDepth );

         // remove the finally point...
         m_finallyStack.pop();

         // this will push the finally handler AND
         // the action to be done when the handler completes.
         check.handleFinally( this, finallyHandler );
         return e_unroll_suspended;
      }

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

   inline void onCrossFinally( VMContext* ) const {}

   inline void onBaseFound( VMContext* ) const {}

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

   inline void onBaseFound( VMContext* ctx ) const {
      // pop the base loop entity and set success
      ctx->popCode();
      ctx->pushData(Item());
   }

   inline void onCrossFinally( VMContext* ) const {}

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
      if( ps.isTry() )
      {
         const StmtTry* stry = static_cast<const StmtTry*>( &ps );
         if( stry->catchSelect().arity() == 0 && stry->catchSelect().getDefault() == 0 )
         {
            // catch all, but do nothing.
            ctx->setCatchBlock( 0 );
            return true;
         }

         SynTree* st = stry->catchSelect().findBlockForType( m_item, ctx );
         ctx->setCatchBlock( st );
         return st != 0;
      }

      return false;
   }


   inline void onCrossFinally( VMContext* ) const {}

   inline bool dontCrossFrame() const { return false; }

   inline void onBaseFound( VMContext* ) const {}

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_raiseTop;

      ctx->pushData( m_item );
      ctx->pushCodeWithUnrollPoint( ps );
      ctx->pushCode( handler );
   }

private:
   Item m_item;
};

class CheckIfCodeIsCatchError
{
public:
   CheckIfCodeIsCatchError( Error* err ):
      m_error(err),
      m_errClass(err->handler())
   {}

   inline bool operator()( const PStep& ps, VMContext* ctx ) const
   {
      if( ps.isTry()  )
      {
         if( ps.isTracedCatch() )
         {
            ctx->setCatchBlock( static_cast<const SynTree*>(&ps) );
            return true;
         }

         const StmtTry* stry = static_cast<const StmtTry*>( &ps );

         if( stry->catchSelect().arity() == 0 && stry->catchSelect().getDefault() == 0 )
         {
            // catch all, but do nothing.
            ctx->setCatchBlock( 0 );
            return true;
         }

         SynTree* st = stry->catchSelect().findBlockForType( Item( m_errClass, m_error), ctx );

         if( st != 0 )
         {
            // found, shall we add a traceback to the error?
            if( st->isTracedCatch() && ! m_error->hasTraceback() )
            {
               ctx->addTrace(m_error);
            }
            ctx->setCatchBlock( st );
            return true;
         }
      }

      return false;
   }

   inline void onCrossFinally( VMContext* ctx ) const
   {
      if( ! m_error->hasTraceback() )
      {
         ctx->addTrace( m_error );
      }
   }

   inline bool dontCrossFrame() const { return false; }

   inline void onBaseFound( VMContext* ) const {}

   inline void handleFinally( VMContext* ctx, const TreeStep* handler ) const
   {
      static PStep* ps = &Engine::instance()->stdSteps()->m_raiseTop;

      ctx->pushData( Item(ctx->thrownError()->handler(), ctx->thrownError()) );
      ctx->pushCodeWithUnrollPoint( ps );
      ctx->pushCode( handler );

   }

private:
   Error* m_error;
   const Class* m_errClass;
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
      if( m_catchBlock != 0 )
      {
         // change the try with the catch
         resetCode( m_catchBlock );
         const Symbol* sym = m_catchBlock->target();
         if( sym != 0 )
         {
            *resolveSymbol(sym, true) = m_raised;
         }
         // be sure to reset all the step-in-yield hierarchy
         atomicOr( m_events, evtEmerge );
      }
      else {
         // we just discarded the item.
         popCode(); // remove the item
         m_raised.setNil();
      }

   }
   else if( result == e_unroll_not_found )
   {
      // reset the raised object, anyhow.
      m_raised.setNil();

      UncaughtError* ce = new UncaughtError( ErrorParam( e_uncaught, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm));
      ce->raised( item );
      raiseError( ce );
   }
   else {
      // be sure to reset all the step-in-yield hierarchy
      atomicOr( m_events, evtEmerge );
      m_raised.setNil();
   }
}

Error* VMContext::raiseError( Error* ce )
{
   // be sure to release the critical section no matter what
   releaseAcquired();

   if( m_lastRaised != 0 ) m_lastRaised->decref();
   m_lastRaised = ce;
   ce->incref();
   if( ce->mantra().empty() )
   {
      contextualize(ce);
   }

   // can we catch it?
   m_catchBlock = 0;
   CheckIfCodeIsCatchError check( ce );

   VMContext::t_unrollResult result = unrollToNext<CheckIfCodeIsCatchError>( check );
   if( result == e_unroll_found )
   {
      if( m_catchBlock != 0 )
      {
         resetCode( m_catchBlock );

         // assign the error to the required item.
         if( m_catchBlock->target() != 0 )
         {
            *resolveSymbol(m_catchBlock->target(), true ) = Item( ce->handler(), ce );
         }
         // be sure to reset all the step-in-yield hierarchy
         atomicOr( m_events, evtEmerge );
      }
      else {
         // discad the error and exit from the try
         m_lastRaised = 0;
         ce->decref();
         popCode();
      }
   }
   else if( result == e_unroll_not_found )
   {
      // we're out of business; ask to raise to our parent.
      atomicOr( m_events, evtRaise );
      // add a trace if not present
      if( ! ce->hasTraceback() )
      {
         addTrace(ce);
      }
   }
   // otherwise, the throw is suspended
   else {
      atomicAnd( m_events, ~evtRaise );
      // be sure to reset all the step-in-yield hierarchy
      atomicOr( m_events, evtEmerge );
      ce->decref();
      m_lastRaised = 0;
   }

   return ce;
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
         ctx->setTerminateEvent();
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


void VMContext::callInternal( const Item& item, int nparams )
{
   TRACE( "Calling item %s -- call frame code:%p, data:%p, call:%p",
         item.describe().c_ize(),
         m_codeStack.m_top, m_dataStack.m_top, m_callStack.m_top  );

   Class* cls = 0;
   void* data = 0;
   item.forceClassInst(cls,data);
   cls->op_call(this, nparams, data );
}

void VMContext::callInternal( Function* function, int nparams, const Item& self )
{
   TRACE( "Calling method %s.%s -- call frame code:%p, data:%p, call:%p",
         (self.isUser() ? self.asClass()->name().c_ize() : "<flat class>"), function->locate().c_ize(),
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
         m_callStack.m_top->m_dataBase );


   makeCallFrame( function, nparams );
   TRACE3( "-- codebase:%d, dynsBase:%d, stackBase:%d",
         m_callStack.m_top->m_codeBase,
         m_callStack.m_top->m_dynsBase,
         m_callStack.m_top->m_dataBase );

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
      m_callStack.m_top->m_dataBase );

   makeCallFrame( closure, nparams );
   // define all the closed parameters.
   ClosedData* cd = closure->data();
   cd->defineSymbols( this );

   TRACE3( "-- codebase:%d, dynsBase:%d, stackBase:%d",
      m_callStack.m_top->m_codeBase,
      m_callStack.m_top->m_dynsBase,
      m_callStack.m_top->m_dataBase );

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
      memcpy( m_dataStack.m_top-pcount+1, params, pcount * sizeof(item) );
   }

   m_caller = currentCode().m_step;
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
      memcpy( m_dataStack.m_top-pcount+1, params, pcount * sizeof(Item) );
   }
   m_caller = currentCode().m_step;
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
      memcpy( m_dataStack.m_top-pcount+1, params, pcount * sizeof(Item) );
   }
   m_caller = currentCode().m_step;
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
      memcpy( m_dataStack.m_top-pcount+1, params, pcount * sizeof(Item) );
   }
   m_caller = currentCode().m_step;
   makeCallFrame(cls, pcount );
   cls->closed()->invoke(this, pcount);
}


void VMContext::addLocalFrame( SymbolMap* st, int pcount )
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   static const Symbol* base = Engine::instance()->baseSymbol();

   if( st != 0 ) {
      TRACE("Add local frame PCOUNT: %d/%d", pcount, st->size() );
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
   baseDyn->m_value = m_dataStack.m_top-pcount;

   // if we don't have a map, there's nothing else we should do.
   if( st == 0 ) {
      popData( pcount + 1 );
      return;
   }

   // Assign the parameters
   Item* top = &topData() - pcount+1;
   int32 p = 0;
   if( pcount > (int32)st->size() )
   {
      pcount = st->size();
   }
   while( p < pcount )
   {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = st->getById(p);
      dd->m_value = top;
      ++top;
      ++p;
   }

   // blank unused parameters
   pcount = st->size();
   while( p < pcount ) {
      DynsData* dd = m_dynsStack.addSlot();
      dd->m_sym = st->getById(p);
      dd->m_value = m_itemStack->push(m_dynsStack.depth());
      dd->m_value->setNil();
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
            static PStep* pop = &Engine::instance()->stdSteps()->m_pop;

            const TreeStep* fd = m_finallyStack.m_top->m_finstep;
            m_finallyStack.pop();
            m_codeStack.m_top = top;
            if( exec ) {
               // chance into the exec frame, keep depth in m_seq
               m_codeStack.m_top->m_step = localFrameExec;
            }
            pushCode( pop );
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
      // clean 1 for the data created by the finally controller.
      currentCode().m_seqId = 1;
      pushCode(fd);
      return;
   }

   // reset code and data
   m_codeStack.unroll( topCall->m_codeBase );
   PARANOID( "Code stack underflow at return", (m_codeStack.m_top >= m_codeStack.m_base-1) );
   // Use initBase as stackBase may have been moved -- but keep 1 parameter ...

   m_dataStack.unroll( topCall->m_dataBase );
   // notice: data stack can never be empty, an empty data stack is an error.
   PARANOID( "Data stack underflow at return", (m_dataStack.m_top >= m_dataStack.m_base) );

   m_dynsStack.unroll( topCall->m_dynsBase );
   m_itemStack->freeUpToDepth(topCall->m_dynsBase);
   PARANOID( "Dynamic Symbols stack underflow at return", (m_dynsStack.m_top >= m_dynsStack.m_base-1) );

   // Forward the return value
   *m_dataStack.m_top = value;

   // Finalize return -- pop call frame
   m_callStack.m_top--;

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
      ctx->topData().setDoubt();
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
      ctx->topData().setDoubt();
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

void VMContext::returnFrameDoubt( const Item& value )
{
   returnFrame_base<ReturnerND>(value);
}

void VMContext::returnFrameEval( const Item& value )
{
   returnFrame_base<ReturnerEval>(value);
}

void VMContext::returnFrameDoubtEval( const Item& value )
{
   returnFrame_base<ReturnerNDEval>(value);
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


void VMContext::defineSymbol( const Symbol* sym, Item* data )
{
   DynsData* newData = m_dynsStack.addSlot();
   newData->m_sym = sym;
   newData->m_value = data;
}


void VMContext::defineSymbol(const Symbol* sym)
{
   DynsData* newData = m_dynsStack.addSlot();
   newData->m_sym = sym;
   newData->m_value = m_itemStack->push( m_dynsStack.depth() );
   newData->m_value->setNil();
}

Item* VMContext::resolveSymbol( const String& symname, bool forAssign )
{
   const Symbol* sym = Engine::getSymbol(symname);
   try
   {
      Item* res = resolveSymbol( sym, forAssign );
      sym->decref();
      return res;
   }
   catch( ... )
   {
      sym->decref();
      throw;
   }
   return 0;
}


Item* VMContext::resolveGlobal( const String& symname, bool forAssign )
{
   const Symbol* sym = Engine::getSymbol(symname);
   try
   {
      Item* res = resolveGlobal( sym, forAssign );
      sym->decref();
      return res;
   }
   catch( ... )
   {
      sym->decref();
      throw;
   }
   return 0;
}


Item* VMContext::resolveSymbol( const Symbol* dyns, bool forAssign )
{
   TRACE1( "VMContext::resolveSymbol -- resolving symbol \"%s\"%s",
            dyns->name().c_ize(),
            (forAssign ? " (for assign)": " (for access)") );

   static const Symbol* baseSym = Engine::instance()->baseSymbol();

   bool isRule = false;

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
         TRACE2( "VMContext::resolveSymbol -- \"%s\" already resolved as %p%s", dyns->name().c_ize(), dd->m_value, (isRule? " (in rule)": "") );

         // did we cross a rule frame?
         if( isRule )
         {
            // -- in this case, we want a copy of the thing we found below.
            DynsData* newSlot = m_dynsStack.addSlot();
            newSlot->m_sym = dyns;
            newSlot->m_value = m_itemStack->push(m_dynsStack.depth(),*dd->m_value);
            return newSlot->m_value;
         }

         return dd->m_value;
      }

      // arrived at a local base?
      if ( baseSym == dd->m_sym )
      {
         if( forAssign )
         {

            // in the end, we must create a new assignable slot.
            // notice that evaluation parameters are above the base symbol.
            DynsData* newSlot = m_dynsStack.addSlot();
            newSlot->m_sym = dyns;
            newSlot->m_value = m_itemStack->push(m_dynsStack.depth());
            newSlot->m_value->setNil();
            TRACE2( "VMContext::resolveSymbol -- \"%s\" went down to a local base, creating new.", dyns->name().c_ize() );
            return newSlot->m_value;
         }

         // this might be a rule base. If it is, we have things to copy below.
         if( dd->m_value->type() == FLC_ITEM_FRAMING) {
            isRule = true;
         }
      }

      --dd;
   }

   // we didn't find it in the local frame. If forassign is true, we must create a new local.
   if( forAssign )
   {
      DynsData* newSlot = m_dynsStack.addSlot();
      newSlot->m_sym = dyns;

      if( currentFrame().m_function->isMain() )
      {
         TRACE2( "VMContext::resolveSymbol -- \"%s\" went down to global context, searching global.", dyns->name().c_ize() );
         newSlot->m_value = resolveGlobal(dyns, true);
         if( newSlot->m_value == 0 )
         {
            TRACE2( "VMContext::resolveSymbol -- \"%s\" NOT found global.", dyns->name().c_ize() );
            newSlot->m_value = m_itemStack->push(m_dynsStack.depth());
            newSlot->m_value->setNil();
         }
      }
      else {
         // in the end, we must create a new assignable slot.
         newSlot->m_value = m_itemStack->push(m_dynsStack.depth());
         newSlot->m_value->setNil();
         TRACE2( "VMContext::resolveSymbol -- \"%s\" went down to function base, creating new.", dyns->name().c_ize() );
      }
      return newSlot->m_value;
   }

   // not found and not for assign; we now must go deep and search it in the call stack,
   // but the logic changes.
   dbase = m_dynsStack.m_base;
   while( dd >= dbase )
   {
      if ( dyns == dd->m_sym ) {
         TRACE2( "VMContext::resolveSymbol -- \"%s\" found in previous frames as %p%s", dyns->name().c_ize(), dd->m_value, (isRule? " (in rule)": "") );
         if( isRule )
         {
            // -- in this case, we want a copy of the thing we found below.
            DynsData* newSlot = m_dynsStack.addSlot();
            newSlot->m_sym = dyns;
            newSlot->m_value = m_itemStack->push(m_dynsStack.depth(),*dd->m_value);
            return newSlot->m_value;
         }

         return dd->m_value;
      }

      --dd;
   }

   // not a local symbol. Try to see if it's global.
   Item* var = resolveGlobal( dyns, forAssign );

   // global arena failed as well. We're doomed
   if( var == 0 )
   {

      Function* func = cf->m_function;
      fassert( func != 0 );
      Module* mod = func->module();
      throw new CodeError( ErrorParam(e_undef_sym, __LINE__, SRC )
               .line(currentCode().m_step->sr().line())
               .module( mod != 0 ? mod->name() : "" )
               .symbol( func->name() )
               .extra(dyns->name())
               );
   }

   DynsData* newSlot = m_dynsStack.addSlot();
   newSlot->m_sym = dyns;

   if( isRule )
   {
      newSlot->m_value = m_itemStack->push(m_dynsStack.depth(),*var);
   }
   else
   {
      newSlot->m_value = var;
   }

   TRACE2( "VMContext::resolveSymbol -- \"%s\" Resolved as new/global/extern %p (%s)%s",
            dyns->name().c_ize(), var, var->describe().c_ize(),
            (isRule? " as rule" : "") );

   return var;
}


Item* VMContext::resolveGlobal( const Symbol* sym, bool forAssign )
{
   TRACE1( "VMContext::resolveGlobal -- resolving %s%s", sym->name().c_ize(), (forAssign? " (for assign)": "" ) )

   // Get the topmost function having a module.
   CallFrame& cf = currentFrame();
   Function* func = cf.m_function;

   Module* mod = 0;
   CallFrame* curFrame = m_callStack.m_top;
   do
   {
      mod = curFrame->m_function->module();
      curFrame--;
   }
   while ( curFrame > m_callStack.m_base && mod == 0  );

   // do we have a module?
   if( mod != 0 )
   {
      // findGlobal will find also externally resolved variables.
      Item* global = mod->resolve( sym );
      if( global != 0 )
      {
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
         Item* global = caller->module()->resolve( sym );
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
      Item* item = mod != 0 ?
         mod->resolve( sym )
         :  process()->modSpace()->findExportedValue( sym ) ;
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


void VMContext::terminate()
{
   setTerminateEvent();

   m_mtx_sleep.lock();
   if( m_bSleeping )
   {
      m_mtx_sleep.unlock();
      // in the meanwhile we might get waken up,
      // but this message is a no-op in that case.
      vm()->contextManager().wakeUp(this);
   }
   else {
      m_mtx_sleep.unlock();
   }
}


void VMContext::onComplete()
{
   onTerminated();
}

void VMContext::onTerminated()
{
   // declare the context dead
   setStatus(statusTerminated);

   // be sure to release any acquired resource.
   // If terminated after a raise, this is a no-op.
   releaseAcquired();

   m_process->onContextTerminated( this );

   // invoke the on termination callbacks.
   // no need to lock, we're supposed to alter this list from the same thread
   m_mtxWeakRef.lock();
   WeakRef* refs = m_firstWeakRef;
   // clear the list, because we might be reset and reused.
   m_firstWeakRef = 0;
   while( refs != 0 )
   {
      refs->terminated( this );
      refs = refs->m_next;
   }
   m_mtxWeakRef.unlock();

   // get the events now
   int value;
   value = atomicFetch( m_events );

   // And then ask the context manager to work on us.
   vm()->contextManager().onContextTerminated(this);

   if( (value & evtRaise) )
   {
      if( m_inGroup != 0 ) {
         m_inGroup->setError(this->m_lastRaised);
         m_inGroup->onContextTerminated(this);
      }
      else {
         // we're the main context.
         m_process->onCompletedWithError(this->m_lastRaised);
      }
   }
   else {
      if( m_inGroup != 0 ) {
         m_inGroup->onContextTerminated(this);
      }
      else {
         // we're the main context.
         m_process->setResult( topData() );
         m_process->onCompleted();
      }
   }

   // relaunched?
   if( getStatus() != statusTerminated )
   {
      return;
   }

   // we're off from the collector...
   Engine::collector()->unregisterContext(this);

}


Error* VMContext::runtimeError( int id, const String& extra, int line )
{
   String noname;
   Function* curFunc = currentFrame().m_function;
   const String* modName = curFunc->module() == 0 ? &noname : &curFunc->module()->name();
   const String* modPath = curFunc->module() == 0 ? &noname : &curFunc->module()->uri();

   if( line == 0 ) {
      line = currentCode().m_step->sr().line();
   }

   CodeError* error = new CodeError( ErrorParam(id, line, *modName )
            .origin(ErrorParam::e_orig_runtime)
            .path(*modPath)
            .symbol( curFunc->name() )
            .extra( extra ) );

   return error;
}

void VMContext::contextualize( Error* error, bool force )
{
   String noname;
   Function* curFunc = currentFrame().m_function;

   if( error->line() == 0 || force )
   {
      CodeFrame* top = m_codeStack.m_top;
      int l = top->m_step->sr().line();
      error->line(l);
   }

   if( error->mantra().empty() || force)
   {
      error->mantra( curFunc->fullName() );
   }

   Module* mod = curFunc->module();
   if( mod != 0 || force )
   {
      if( error->module().empty() )
      {
         error->module(mod->name());
      }

      if( error->path().empty() )
      {
         error->path(mod->uri());
      }

      if( error->handler() == 0 )
      {
         // try to contextualize the handler in the module
         const Class* cls = mod->getClass(error->className());
         if( cls != 0 )
         {
            error->handler(cls);
         }
      }
   }

}

void VMContext::addTrace( Error *error )
{
   VMContext* ctx = this;

   long depth = ctx->callDepth();
   const PStep* caller = currentCode().m_step;

   for( long i = 0; i < depth; ++i )
   {
      CallFrame& cf = ctx->callerFrame(i);
      Function* func = cf.m_function;
      Module* mod = func->module();

      if( caller == 0 )
      {
         caller = ctx->nextStep(i);
      }

      int line = caller != 0 ? caller->line() : func->declaredAt();
      if( mod != 0 )
      {
         error->addTrace( TraceStep(mod->name(), mod->uri(), func->fullName(), line ) );
      }
      else
      {
         error->addTrace( TraceStep("<internal>", func->fullName(), line ) );
      }

      caller = cf.m_caller;
   }
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
         // TODO: Mark the pstep if it's a treestep.
         // actually, it might be a bit paranoid.
         ++base;
      }
   }

   // then, other various elements.
   {
      m_initWrite.gcMark(mark);
      m_initRead.gcMark(mark);
      //TODO: let the collector do this property
      //process()->modSpace()->gcMark(mark);
   }
}

void VMContext::swapOut()
{
   setCompleteEvent();
   process()->removeLiveContext(this);
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

GCToken* VMContext::addNewToken( GCToken* token )
{
   token->m_next = m_newTokens->m_next;
   token->m_prev = m_newTokens;

   m_newTokens->m_next->m_prev = token;
   m_newTokens->m_next = token;
   return token;
}


void VMContext::getNewTokens( GCToken* &first, GCToken* &last )
{
   if( m_newTokens->m_next == m_newTokens )
   {
      first = 0;
      last = 0;
      return;
   }

   first = m_newTokens->m_next;
   last = m_newTokens->m_prev;

   m_newTokens->m_next = m_newTokens;
   m_newTokens->m_prev = m_newTokens;

   // Not necessary
   first->m_prev = 0;
   last->m_next = 0;
}


int VMContext::getStatus()
{
   return atomicFetch(m_status);
}

void VMContext::setStatus( int status )
{
   atomicSet(m_status, status);
}

//===============================================================
// Fill an error with the current context.
//
ErrorParam::ErrorParam( int code, VMContext* ctx, const char* file, int signLine )
{
   String noname;
   Function* curFunc = ctx->currentFrame().m_function;
   const String* modName = curFunc->module() == 0 ? &noname : &curFunc->module()->name();
   const String* modPath = curFunc->module() == 0 ? &noname : &curFunc->module()->uri();

   m_errorCode = code;
   m_line = ctx->currentCode().m_step->sr().line();
   m_module = *modName;
   m_path = *modPath;
   m_symbol = curFunc->fullName();
   m_origin = e_orig_script;

   if( file != 0 )
   {
      m_signature = file;
      if( signLine != 0 ) {
         m_signature.A(":").N(signLine);
      }
   }

   m_sysError = 0;
   m_catchable = true;
   m_chr = 0;
}

}

/* end of vmcontext.cpp */
