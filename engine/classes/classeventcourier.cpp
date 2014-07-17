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

   ctx->stepIn(cevt->stepEngage());
}


class PStepAfterHandlingCatch: public StmtTry
{
public:
   PStepAfterHandlingCatch() {
      apply = apply_;
   }

   virtual ~PStepAfterHandlingCatch() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepAfterHandlingFinally";
   }


   static void apply_( const PStep*, VMContext* ctx )
   {
      MESSAGE2("Entering ClassEventCourier::PStepAfterHandlingFinally");

      fassert( String("EventCourier::Token") == ctx->topData().asOpaqueName());
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(ctx->topData().asOpaque());
      tk->aborted(ctx->thrownError());
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

      fassert( String("EventCourier::Token") == ctx->topData().asOpaqueName());
      EventCourier::Token* tk = static_cast<EventCourier::Token*>(ctx->topData().asOpaque());
      tk->completed(retval);
      tk->decref();
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


class PStepEngage: public PStep
{
public:
   PStepEngage() { apply = apply_; }
   virtual ~PStepEngage() {}
   virtual void describeTo( String& target ) const
   {
      target = "ClassEventCourier::PStepEngage";
   }

   static void apply_( const PStep*, VMContext* ctx )
   {
      static ClassEventCourier* cls =
               static_cast<ClassEventCourier*>(Engine::instance()->stdHandlers()->eventCourierClass());


      // the courier object is the self of our frame.
      EventCourier* courier = ctx->tself<EventCourier*>();
      EventCourier::Token* tk = courier->popEvent();
      TRACE2("Entering ClassEventCourier::PStepAfterWait courier: %p token%p", courier, tk);

      // no event for us?
      if( tk == 0 )
      {
         // wait for more -- how long? -- the parameter is in our frame.
         Item* i_to = ctx->param(0);
         int64 to = i_to == 0 || i_to->isNil() ? 0 :  i_to->forceInteger();

         // we'll wait on the event.
         ctx->pushCode( cls->stepAfterHandling() );
         ctx->pushCodeWithUnrollPoint( cls->stepAfterHandlingCatch() );

         ctx->addWait( courier->eventPosted() );
         ctx->engageWait(to);
      }
      else
      {
         // have we got an event handler...
         String message;
         const Item* cb = courier->getHandler( tk->eventID(), message );
         if( cb == 0 )
         {
            // or a generic handler...
            cb = courier->getDefaultHandler();
         }

         // we have something to do
         if( cb != 0 )
         {
            ctx->pushCode( cls->stepAfterHandling() );

            // we push the token in the stack to have a nice place where to get it
            ctx->pushData( Item().setOpaque("EventCourier::Token", tk) );

            // compose the call directly
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
            ctx->callInternal(*cb, (int)tk->params().length() );
         }
         else {
            // can throw
            courier->onUnhandled(tk, ctx);
            // in any case, don't pop
         }
      }
   }
};

}

ClassEventCourier::ClassEventCourier():
         Class("EventCourier")
{
   m_funcWait = new FALCON_FUNCTION_NAME(engage);
   m_stepEngage = new PStepEngage;
   m_stepAfterWait = new PStepAfterWait;
   m_stepAfterHandling = new PStepAfterHandling;
   //m_stepAfterHandlingCatch =  new PStepAfterHandlingCatch;
   m_tokenClass = new ClassToken;

   addMethod(m_funcWait);
}

ClassEventCourier::~ClassEventCourier()
{
   delete m_tokenClass;
   delete m_stepAfterHandlingCatch;
   delete m_stepEngage;
   delete m_stepAfterWait;
   delete m_stepAfterHandling;
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
