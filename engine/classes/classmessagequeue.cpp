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



ClassMessageQueue::ClassMessageQueue():
      ClassShared("MessageQueue"),
      FALCON_INIT_PROPERTY(subscribers),
      FALCON_INIT_PROPERTY(empty),

      FALCON_INIT_METHOD(send),
      FALCON_INIT_METHOD(sendEvent),
      FALCON_INIT_METHOD(marshal),
      FALCON_INIT_METHOD(get),
      FALCON_INIT_METHOD(peek),
      FALCON_INIT_METHOD(subscribersFence),
      FALCON_INIT_METHOD(tryWait),
      FALCON_INIT_METHOD(wait),
      FALCON_INIT_METHOD(subscribe),
      FALCON_INIT_METHOD(unsubscribe)
{
   static Class* shared = Engine::handlers()->sharedClass();
   addParent(shared);
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


FALCON_DEFINE_PROPERTY_GET_P( ClassMessageQueue, subscribers )
{
   MessageQueue* self = static_cast<MessageQueue*>(instance);
   value = (int64) self->subscribers();
}

FALCON_DEFINE_PROPERTY_SET_P0( ClassMessageQueue, subscribers )
{
   throw readOnlyError();
}

FALCON_DEFINE_PROPERTY_GET_P( ClassMessageQueue, empty )
{
   MessageQueue* self = static_cast<MessageQueue*>(instance);
   value.setBoolean( self->consumeSignal(Processor::currentProcessor()->currentContext(), 1) == 0 );
}

FALCON_DEFINE_PROPERTY_SET_P0( ClassMessageQueue, empty )
{
   throw readOnlyError();
}


FALCON_DEFINE_METHOD_P( ClassMessageQueue, send )
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


FALCON_DEFINE_METHOD_P( ClassMessageQueue, sendEvent )
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


FALCON_DEFINE_METHOD_P( ClassMessageQueue, marshal )
{
   if( pCount < 1 )
   {
      throw paramError(__LINE__, SRC );
   }

   EventMarshal* evm = new EventMarshal(*ctx->param(0));
   ctx->returnFrame( FALCON_GC_HANDLE(evm) );
}


FALCON_DEFINE_METHOD_P1( ClassMessageQueue, get )
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


FALCON_DEFINE_METHOD_P1( ClassMessageQueue, subscribersFence )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   int32 count = 0;
   Item* i_count = ctx->param(0);
   if( i_count == 0 || ! i_count->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }
   else {
      count = i_count->forceInteger();
      if( count <= 0 )
      {
         throw paramError( __LINE__, SRC );
      }
   }

   Shared* sf = self->subscriberWaiter(count);
   // the sf comes already given to the GC by subscriberWaiter(count);
   ctx->returnFrame( Item(sf->handler(), sf) );
}


FALCON_DEFINE_METHOD_P1( ClassMessageQueue, peek )
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


FALCON_DEFINE_METHOD_P( ClassMessageQueue, tryWait )
{
   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}

FALCON_DEFINE_METHOD_P( ClassMessageQueue, wait )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // subscribe (no-op if already subscribed)
   self->subscribe(ctx);

   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}

FALCON_DEFINE_METHOD_P1( ClassMessageQueue, subscribe )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // subscribe (no-op if already subscribed)
   self->subscribe(ctx);
   ctx->returnFrame();
}

FALCON_DEFINE_METHOD_P1( ClassMessageQueue, unsubscribe )
{
   MessageQueue* self = static_cast<MessageQueue*>(ctx->self().asClass()->getParentData(this->methodOf(), ctx->self().asInst()) );

   // unsubscribe (no-op if not subscribed)
   self->unsubscribe(ctx);
   ctx->returnFrame();
}

}

/* end of classmessagequeue.cpp */
