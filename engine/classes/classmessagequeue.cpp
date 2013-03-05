/*
   FALCON - The Falcon Programming Language.
   FILE: classmessagequeue.cpp

   Message queue reflection in scripts
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 20 Feb 2013 14:15:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/class/classmessagequeue.cpp"

#include <falcon/classes/classmessagequeue.h>
#include <falcon/messagequeue.h>
#include <falcon/cm/semaphore.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/processor.h>
#include <falcon/eventmarshal.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/accesserror.h>

#include <falcon/vm.h>

#include <map>

namespace Falcon {

/*#
  @property subscribers MessageQueue
  @brief Number of subscribers currently listening on this queue.

 As the message queue doesn't support late subscribers, a sender
 can safely check for subscribers count to be > 0 before preparing
 a complex and time-consuming message that would be immediately
 discarded by the queue.

 */

static void get_subscribers( const Class*, const String&, void* instance, Item& value )
{
   MessageQueue* self = static_cast<MessageQueue*>(instance);
   value = (int64) self->subscribers();
}


/*#
  @property empty MessageQueue
  @brief True if the queue has not any item that can be read by the calling context.

 */
static void get_empty( const Class*, const String&, void* instance, Item& value )
{
   MessageQueue* self = static_cast<MessageQueue*>(instance);
   value.setBoolean( self->consumeSignal(Processor::currentProcessor()->currentContext(), 1) == 0 );
}

/*#
  @method send MessageQueue
  @brief Sends a new message to all the subscribers
  @optparam message An arbitrary item that will be received by all the subscribers.
 */
FALCON_DECLARE_FUNCTION( send, "message:X,..." );

/*#
  @method sendEvent MessageQueue
  @brief Sends a marshaled event to the listeners.
  @param name An arbitrary name for the event.
  @optparam message An arbitrary item that will be received by all the subscribers.

 */
FALCON_DECLARE_FUNCTION( sendEvent, "event:S,message:X,..." );

/*#
  @method marshal MessageQueue
  @brief Creates a marshal function that can be used in Waiters.
  @param name An arbitrary name for the event.
  @optparam message An arbitrary item that will be received by all the subscribers.

   This method returns a marshal function. The function expects to receive this
   queue as a parameter, and gets the incoming message.
   It then invokes a method on the @b handler object named after the event sent
   by @a MessageQueue.sendEvent, named accordingly to the following rules:
   - If the event is not named (or has been sent via @a MessageQueue.send), then
      a method named "on_"  is searched and invoked.
   - If the event is named, a method called similarly to "on_EventName" is searched.
   - If that method is not found, a method "on__discard" (two underlines) is searched
      and eventually invoked.
   - If that method is not found, a method "on__default" (two underlines) is searched
      and invoked; This method receives the name of the generated event as the first
      parameter.

  If the message was sent via @a MessageQueue.sendEvent, then the method receives the
  parameters as they were originally written in the sendEvent call.

  The "on_" method always receive a single parameter.
 */
FALCON_DECLARE_FUNCTION( marshal, "handler:X" );


/*#
  @method get MessageQueue
  @brief Gets a message waiting on the queue.
  @return the item posted by send()
  @raise AccessError if the queue has not any new message for us.

  Removes the message from the front of the queue.
 */
FALCON_DECLARE_FUNCTION( get, "" );

/*#
  @method peek MessageQueue
  @brief Peek a message waiting on the queue.ClassMessageQueue
  @return the item posted by send()
  @raise AccessError if the queue has not any new message for us.

  Reads the message in front of the queue, without removing it.
 */
FALCON_DECLARE_FUNCTION( peek, "" );

/*#
  @method subscribersFence MessageQueue
  @brief Return a fence that gets notified when the required amount of subscribers have joined the queue.
  @return A waitable fence.

  This facility obviates the need to use other synchronization devices to
  get to know when the subscribers that are expected to join have really
  joined.

  The message queue doesn't implement late-arrival messaging semantic. Messages
  are delivered exclusively to the subscribers that have subscribed the list
  in the moment the message is sent.

  For this reason, before sending critical messages to listeners that might
  be not ready to receive them, it is necessary to ascertain that they have
  actually subscribed. This could be done using other synchronization devices,
  but the subscribersFence() method returns a fence that is automatically
  updated as the receivers subscribe the queue, so no other action is required
  on the subscribers side.
 */
FALCON_DECLARE_FUNCTION( subscribersFence, "count:N" );


/*#
  @method tryWait MessageQueue
  @brief Check if there are new messages for us.
  @return true if there are new messages for us.
 */
FALCON_DECLARE_FUNCTION( tryWait, "" );

/*#
  @method wait MessageQueue
  @brief Wait for new messages to be available.
  @optparam timeout Milliseconds to wait for the barrier to be open.
  @return true if the barrier is open during the wait, false if the given timeout expires.

  If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
  the wait exits immediately.

  @note If not previously subscribed, the receiver gets subscribed
  at the moment the wait is entered.
 */
FALCON_DECLARE_FUNCTION( wait, "timeout:[N]" );

/*#
  @method subscribe MessageQueue
  @brief Explicitly subscribe the currently running context to the message queue.

  This is implicitly done when a first wait() is issued on the queue, but
  the invoker might want to subscriber at a earlier moment and then receive
  the messages that were sent in the meanwhile at a later time.
 */
FALCON_DECLARE_FUNCTION( subscribe, "" );

/*#
  @method unsubscribe MessageQueue
  @brief Unsubscribe the currently running context from the message queue.

  From this moment on, the context will not receive the messages sent to the queue
  anymore, until resubscribed or waiting again on the queue.
 */
FALCON_DECLARE_FUNCTION( unsubscribe, "" );



void Function_send::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount == 0 )
   {
      throw paramError(__LINE__, SRC );
   }

   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );
   for( int32 i = 0;  i < pCount; ++i )
   {
      self->send( *ctx->param(i) );
   }

   ctx->returnFrame();
}


void Function_sendEvent::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount < 2 )
   {
      throw paramError(__LINE__, SRC );
   }

   Item* i_name = ctx->param(0);
   if( ! i_name->isString() )
   {
      throw paramError(__LINE__, SRC );
   }

   // copy
   String eventName = *i_name->asString();
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );
   Item toSend;
   if( pCount == 2 )
   {
      toSend = *ctx->param(1);
   }
   else
   {
      ItemArray* array = new ItemArray;
      for( int32 i = 1; i < pCount; ++i )
      {
         array->append( *ctx->param(i) );
      }
      // mark the string so that the other side knows
      eventName.prepend(' ');
      toSend = FALCON_GC_HANDLE(array);
   }

   self->sendEvent(eventName, toSend);
   ctx->returnFrame();
}


void Function_marshal::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount < 1 )
   {
      throw paramError(__LINE__, SRC );
   }

   EventMarshal* evm = new EventMarshal(*ctx->param(0));
   ctx->returnFrame( FALCON_GC_HANDLE(evm) );
}


void Function_get::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   Item msg;
   if( ! self->get(ctx, msg ) )
   {
      throw new AccessError( ErrorParam( e_acc_forbidden, __LINE__, SRC )
               .extra("Empty queue"));
   }
   ctx->returnFrame( msg );
}


void Function_subscribersFence::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   int32 count = 0;
   Item* i_count = ctx->param(0);
   if( i_count == 0 || ! i_count->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }
   else {
      count = (int32) i_count->forceInteger();
      if( count <= 0 )
      {
         throw paramError( __LINE__, SRC );
      }
   }

   Shared* sf = self->subscriberWaiter(count);
   // the sf comes already given to the GC by subscriberWaiter(count);
   ctx->returnFrame( Item(sf->handler(), sf) );
}


void Function_peek::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   Item msg;
   if( ! self->peek(ctx, msg ) )
   {
      throw new AccessError( ErrorParam( e_acc_forbidden, __LINE__, SRC )
               .extra("Empty queue"));
   }
   ctx->returnFrame( msg );
}


void Function_wait::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // subscribe (no-op if already subscribed)
   self->subscribe(ctx);

   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}


void Function_subscribe::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // subscribe (no-op if already subscribed)
   self->subscribe(ctx);
   ctx->returnFrame();
}

void Function_unsubscribe::invoke(VMContext* ctx, int32 pCount )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // unsubscribe (no-op if not subscribed)
   self->unsubscribe(ctx);
   ctx->returnFrame();
}



ClassMessageQueue::ClassMessageQueue():
      ClassShared("MessageQueue")
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);
   addProperty("empty", get_empty );
   addProperty("subscribers", get_subscribers );

   addMethod( new Function_send );
   addMethod( new Function_sendEvent );
   addMethod( new Function_marshal );
   addMethod( new Function_get );
   addMethod( new Function_subscribersFence );
   addMethod( new Function_peek );
   addMethod( new Function_wait );
   addMethod( new Function_subscribe );
   addMethod( new Function_unsubscribe );

}

ClassMessageQueue::~ClassMessageQueue()
{}

void* ClassMessageQueue::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassMessageQueue::op_init( VMContext* ctx, void*, int pCount ) const
{
   String name;

   if( pCount > 0 )
   {
      Item* i_name = ctx->opcodeParams(pCount);
      if( ! i_name->isString() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("[S]") );
      }

      name = *i_name->asString();
   }

   MessageQueue* msgq;
   if( name != "" )
   {
      msgq = ctx->vm()->getMessageQueue( name );
   }
   else {
      msgq = new MessageQueue(&ctx->vm()->contextManager());
      // VM queues are not subject to garbage collecting.
      // (yet they get marked when reachable).
      FALCON_GC_STORE(this, msgq);
   }

   ctx->stackResult(pCount+1, Item(msgq->handler(),msgq) );
   return true;
}

}

/* end of classmessagequeue.cpp */
