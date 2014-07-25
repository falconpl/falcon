/*
   FALCON - The Falcon Programming Language.
   FILE: classeventcourier.cpp

   Sript/VM interface to the EventCourier message dispatching system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 15 Jul 2014 16:48:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classeventcourier.cpp"

#include <falcon/classes/classeventcourier.h>
#include <falcon/eventcourier.h>
#include <falcon/function.h>
#include <falcon/pstep.h>
#include <falcon/trace.h>
#include <falcon/error.h>
#include <falcon/vmcontext.h>
#include <falcon/psteps/stmttry.h>
#include <falcon/stdhandlers.h>
#include <falcon/shared.h>
#include <falcon/itemarray.h>
#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/classes/classshared.h>

namespace Falcon
{

namespace {

// internal class to handle the tokens in the GC
class ClassToken: public Class
{
public:
   ClassToken():
      Class("EventCourierToken")
   {}

   virtual ~ClassToken() {}

   virtual void dispose( void* instance ) const
   {
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(instance);
      tk->decref();
   }

   virtual void* clone( void* instance ) const
   {
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(instance);
      tk->incref();
      return tk;
   }

   virtual void* createInstance() const
   {
      return 0;
   }

   virtual void gcMarkInstance( void* instance, uint32 mark ) const
   {
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(instance);
      tk->gcMark(mark);
   }


   virtual bool gcCheckInstance( void* instance, uint32 mark ) const
   {
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(instance);
      return tk->currentMark() >= mark;
   }
};

/*# @method engage EventCourier
 @optparam to Maximum timeout before returning.
 @return The value of terminate(), or nil if timed out.

*/

FALCON_DECLARE_FUNCTION(engage, "to:N")
FALCON_DEFINE_FUNCTION_P1(engage)
{
   ClassEventCourier* cevt = static_cast<ClassEventCourier*>(this->methodOf());

   // add a local that we need to store the traveling tokens.
   ctx->addLocals(1);

   // TODO: A serious catch-gate
   if( cevt->stepAfterHandling() == 0 )
   {
      cevt->init();
   }

   EventCourier* evt = ctx->tself<EventCourier>();
   evt->createShared( ctx );
   evt->kickIn();

   ctx->stepIn(cevt->stepEngage());
}


/*# @method send EventCourier
 @brief Sends an event and return immediately.
 @param evtID The event ID to be sent.
 @optparam ... Parameters to be sent to the event.
*/
FALCON_DECLARE_FUNCTION(send, "evtID:N,...")
FALCON_DEFINE_FUNCTION_P(send)
{
   EventCourier* evt = ctx->tself<EventCourier>();
   int64 evtID;
   if( ! FALCON_NPCHECK_GET(0,Ordinal,evtID) )
   {
      throw paramError(__LINE__, SRC);
   }

   evt->sendEvent(evtID, ctx->params()+1, pCount-1 );
   ctx->returnFrame();
}



/*# @method sendWait EventCourier
 @brief Sends an event and wait for the event to be completed
 @param evtID The event ID to be sent.
 @optparam ... Parameters to be sent to the event.
 @return The return value of the event handler
 @raise An error if the event handler raised an error.
*/
FALCON_DECLARE_FUNCTION(sendWait, "evtID:N,...")
FALCON_DEFINE_FUNCTION_P(sendWait)
{
   ClassEventCourier* cevt = static_cast<ClassEventCourier*>(this->methodOf());

   EventCourier* evt = ctx->tself<EventCourier>();
   int64 evtID;
   if( ! FALCON_NPCHECK_GET(0,Ordinal,evtID) )
   {
      throw paramError(__LINE__, SRC);
   }

   Shared* shared = new Shared(&ctx->vm()->contextManager(), Engine::instance()->stdHandlers()->sharedClass());

   ctx->addLocals(2);
   EventCourier::Token* tk = evt->sendEvent(evtID, ctx->params()+1, pCount-1, shared );
   *ctx->local(0) = FALCON_GC_STORE(cevt->tokenClass(), tk);
   *ctx->local(1) = FALCON_GC_HANDLE(shared);

   // ...and a handler for the normal operation
   ctx->pushCode( cevt->stepAfterSendWait() );

   // and engage the wait
   ctx->addWait(shared);
   ctx->engageWait(-1);

   // do not return
}


/*# @method subscribe EventCourier
 @param evtID A numeric event ID to be handled.
 @param handler the object or code handling the event.
 @optparam message A summon message to be sent to the given handler
 @return The self object

 If handler is @b nil, the message is unsubscribed.

 If @b message is given, the handler won't be directly invoked;
 instead, it will be summoned as if invoked with
 @code
    handler::message[...]
 @endocde

 This allows to set or reset plain propeties, or delegate the summoning
 to other objects without changing the event handler.
*/

FALCON_DECLARE_FUNCTION(subscribe, "evtID:N,handler:X,message:[S]")
FALCON_DEFINE_FUNCTION_P1(subscribe)
{
   int64 evtID = 0;
   Item* i_handler = 0;
   String* msg = 0;
   String dfltMsg;

   if ( ! (
            FALCON_NPCHECK_GET(0,Integer, evtID )
            && (i_handler = ctx->param(1)) != 0
            && FALCON_NPCHECK_O_GET(2,String,msg, &dfltMsg)
            )
   )
   {
      throw paramError(__LINE__, SRC);
   }

   EventCourier* evtc = ctx->tself<EventCourier>();
   if( i_handler->isNil() )
   {
      evtc->clearCallback(evtID);
   }
   else {
      evtc->setCallback(evtID, *i_handler, *msg);
   }
   ctx->returnFrame(ctx->self());
}


/*# @method onUnknown EventCourier
 *
 @param handler the object or code handling the default event.
 @optparam message A summon message to be sent to the given handler
 @return The self object

 If @b handler is @b nil, the event handler is cleared.
*/

FALCON_DECLARE_FUNCTION(onUnknown, "handler:X,message:[S]")
FALCON_DEFINE_FUNCTION_P1(onUnknown)
{
   Item* i_handler = 0;
   String* msg = 0;
   String dfltMsg;

   if ( ! (
            (i_handler = ctx->param(0)) != 0
            && FALCON_NPCHECK_O_GET(1,String,msg, &dfltMsg)
            )
   )
   {
      throw paramError(__LINE__, SRC);
   }

   EventCourier* evtc = ctx->tself<EventCourier>();
   if( i_handler->isNil() )
   {
      evtc->clearDefaultCallback();
   }
   else {
      evtc->setDefaultCallback( *i_handler, *msg);
   }
   ctx->returnFrame(ctx->self());
}


class PStepAfterHandlingCatch: public StmtTry
{
public:
   PStepAfterHandlingCatch()
   {
      TreeStep* dflt = new SynTree;
      dflt->apply = apply_;
      catchSelect().append( dflt );
   }

   virtual ~PStepAfterHandlingCatch() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepAfterHandlingFinally";
   }


   static void apply_( const PStep*, VMContext* ctx )
   {
      MESSAGE2("Entering ClassEventCourier::PStepAfterHandlingCatch");

      EventCourier::Token* tk = static_cast<EventCourier::Token*>(ctx->local(0)->asInst());
      Error* error;
      if ( ctx->thrownError() == 0 )
      {
         // we have a raised item
         UncaughtError* ce = new UncaughtError( ErrorParam( e_uncaught, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_vm));
         ce->raised( ctx->raised() );
         error = ce;
         tk->aborted(error);
      }
      else {
         error = ctx->thrownError();
         tk->aborted(error);
         error->decref();
      }

      // we have now an extra reference
      tk->decref();
      ctx->popCode();
   }
};


class PStepAfterHandling: public PStep
{
public:
   PStepAfterHandling() {
      apply = apply_;
   }

   virtual ~PStepAfterHandling() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepAfterHandling";
   }


   static void apply_( const PStep*, VMContext* ctx )
   {
      MESSAGE2("Entering ClassEventCourier::PStepAfterHandling");

      Item retval = ctx->topData();
      ctx->popData();

      EventCourier::Token* tk = static_cast<EventCourier::Token*>(ctx->local(0)->asInst());
      tk->completed(retval);
      tk->decref();
      // kill also the catch below us.
      ctx->popCode(2);
   }
};


class PStepAfterWait: public PStep
{
public:
   PStepAfterWait() { apply = apply_; }
   virtual ~PStepAfterWait() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepAfterWait";
   }


   static void apply_( const PStep*, VMContext* ctx )
   {
      MESSAGE2("Entering ClassEventCourier::PStepAfterWait");

      // did we have a shared resource signaled?
      Shared* shared = ctx->getSignaledResouce();
      if (shared == 0)
      {
         // we timed out -- exit from the function
         ctx->returnFrameDoubt(Item());
      }
      else
      {
         // give the control back to PStepEngage.
         shared->decref();
         ctx->popCode();
      }
   }
};

class PStepAfterSendWait: public PStep
{
public:
   PStepAfterSendWait() { apply = apply_; }
   virtual ~PStepAfterSendWait() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepAfterSendWait";
   }


   static void apply_( const PStep*, VMContext* ctx )
   {
      MESSAGE2("Entering ClassEventCourier::PStepAfterSendWait");

      EventCourier::Token* tk = static_cast<EventCourier::Token*>(ctx->local(0)->asInst());
      if( tk->error() )
      {
         Error* error = tk->error();
         tk->decref();
         throw error;
      }
      else {
         ctx->returnFrame(tk->result());
         tk->decref();
      }
   }
};


class PStepEngage: public PStep
{
public:
   PStepEngage(ClassEventCourier* owner):m_owner(owner) { apply = apply_; }
   virtual ~PStepEngage() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepEngage";
   }

   static void apply_( const PStep* ps, VMContext* ctx )
   {
      static ClassEventCourier* cls =
               static_cast<ClassEventCourier*>(Engine::instance()->stdHandlers()->eventCourierClass());

      const PStepEngage* self = static_cast<const PStepEngage*>(ps);

      // the courier object is the self of our frame.
      EventCourier* courier = ctx->tself<EventCourier>();
      EventCourier::Token* tk = courier->popEvent();
      TRACE2("Entering ClassEventCourier::PStepAfterWait courier: %p token%p", courier, tk);

      // no event for us?
      if( tk == 0 )
      {
         // wait for more -- how long? -- the parameter is in our frame.
         Item* i_to = ctx->param(0);
         int64 to = i_to == 0 || i_to->isNil() ? 0 :  i_to->forceInteger();

         // we'll be called back again here.
         ctx->addWait( courier->eventPosted() );
         ctx->engageWait(to);
      }
      else if( tk->isTerminate() )
      {
         tk->decref();
         ctx->returnFrame();
      }
      else
      {
         // have we got an event handler...
         String message;
         const Item* cb = courier->getCallback( tk->eventID(), message );
         if( cb == 0 )
         {
            // or a generic handler...
            cb = courier->getDefaultCallback(message);
         }

         // we have something to do
         if( cb != 0 )
         {
            // Set a gate for catching errors...
            ctx->pushCodeWithUnrollPoint( cls->stepAfterHandlingCatch() );
            // ...and a handler for the normal operation
            ctx->pushCode( cls->stepAfterHandling() );

            // The token is at local (0) in our frame function
            ctx->local(0)->setUser(self->m_owner->tokenClass(), tk);

            // prepare the base item for call internal
            ctx->pushData(*cb);
            // push first the event ID
            ctx->pushData(Item().setInteger(tk->eventID()));
            // then all the parameters
            for( length_t i = 0; i < tk->params().length(); ++i )
            {
               const Item& value = tk->params()[i];
               ctx->pushData(value);
            }

            // perform the internal call.
            if( message.empty() )
            {
               ctx->callInternal(*cb, (int)tk->params().length()+1 );
            }
            else {
               Class* cls = 0;
               void* inst = 0;
               cb->forceClassInst(cls, inst);
               cls->op_summon(ctx, inst, message, (int)tk->params().length()+1, false );
            }
         }
         else {
            // can throw
            courier->onUnhandled(tk, ctx);
            // in any case, don't pop
         }
      }
   }

private:
   ClassEventCourier* m_owner;
};

}

ClassEventCourier::ClassEventCourier():
         Class("EventCourier")
{
   m_funcWait = new FALCON_FUNCTION_NAME(engage);
   addMethod(m_funcWait);
   addMethod(new FALCON_FUNCTION_NAME(subscribe));
   addMethod(new FALCON_FUNCTION_NAME(onUnknown));
   addMethod(new FALCON_FUNCTION_NAME(send));
   addMethod(new FALCON_FUNCTION_NAME(sendWait));
}

void ClassEventCourier::init()
{
   m_stepEngage = new PStepEngage(this);
   m_stepAfterWait = new PStepAfterWait;
   m_stepAfterHandling = new PStepAfterHandling;
   m_stepAfterHandlingCatch = new PStepAfterHandlingCatch;
   m_tokenClass = new ClassToken;
   m_stepAfterSendWait = new PStepAfterSendWait;
}

ClassEventCourier::~ClassEventCourier()
{
   delete m_tokenClass;
   delete m_stepAfterHandlingCatch;
   delete m_stepEngage;
   delete m_stepAfterWait;
   delete m_stepAfterHandling;
   delete m_stepAfterSendWait;
}


void ClassEventCourier::dispose( void* instance ) const
{
   EventCourier* ec = static_cast<EventCourier*>(instance);
   delete ec;
}

void* ClassEventCourier::clone( void* instance ) const
{
   EventCourier* ec = static_cast<EventCourier*>(instance);
   return new EventCourier(*ec);
}

void* ClassEventCourier::createInstance() const
{
   return new EventCourier;
}


void ClassEventCourier::gcMarkInstance( void* instance, uint32 mark ) const
{
   EventCourier* ec = static_cast<EventCourier*>(instance);
   ec->gcMark(mark);
}


bool ClassEventCourier::gcCheckInstance( void* instance, uint32 mark ) const
{
   EventCourier* ec = static_cast<EventCourier*>(instance);
   return ec->currentMark() >= mark;
}

}

/* classeventcourier.cpp */
