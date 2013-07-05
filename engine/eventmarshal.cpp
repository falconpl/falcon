/*
   FALCON - The Falcon Programming Language.
   FILE: eventmarshal.cpp

   A function object specialized in dispatching events from queues
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 21 Feb 2013 08:42:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/eventmarshall.cpp"

#include <falcon/eventmarshal.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/trace.h>
#include <falcon/pstep.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>

#include <falcon/messagequeue.h>
#include <falcon/itemarray.h>
#include <falcon/stderrors.h>


#define MARSHAL_PREFIX "on_"

namespace Falcon
{


EventMarshal::EventMarshal( const Item& marshalled ):
         Function("eventMarshall"),
         m_stepPropResolved(this)
{
   parseDescription("queue:MessageQueue");

   marshalled.forceClassInst(m_cls, m_instance);
}

EventMarshal::EventMarshal( Class* marshallCls, void *marshallInstance ):
   Function("eventMarshall"),
   m_cls(marshallCls),
   m_instance( marshallInstance ),
   m_stepPropResolved(this)
{
   parseDescription("queue:MessageQueue");
}

EventMarshal::~EventMarshal()
{
}


void EventMarshal::invoke( VMContext* ctx, int32 )
{  
   static Class* queueCls = Engine::handlers()->messageQueueClass();

   Item* i_queue = ctx->param(0);
   Class* paramCls = 0;
   void* paramInst = 0;

   if( i_queue == 0 || ! i_queue->asClassInst(paramCls, paramInst ) || ! paramCls->isDerivedFrom(queueCls) )
   {
      throw paramError(__LINE__, SRC);
   }
   
   MessageQueue* mq = static_cast<MessageQueue*>(paramCls->getParentData(queueCls, paramInst));
   String evtName;
   Item message;
   bool result = mq->getEvent(ctx, evtName, message);
   if( ! result )
   {
      throw new AccessError( ErrorParam( e_acc_forbidden, __LINE__, SRC )
               .extra("Empty queue"));
   }

   bool asArray = false;
   if( evtName.size() != 0 && evtName.getCharAt(0) == ' ' )
   {
      evtName = evtName.subString(1);
      asArray = true;
   }

   ctx->addLocals(3);
   *ctx->local(0) = message;
   ctx->local(1)->setBoolean(asArray);

   bool success;
   String property;
   // get the marshalled event.
   if( evtName.size() == 0 )
   {
      property = MARSHAL_PREFIX;
      success = m_cls->hasProperty( m_instance, property );
   }
   // else call op_EventName
   else {
      property = MARSHAL_PREFIX + evtName;
      success = m_cls->hasProperty( m_instance, property );

      if( !success )
      {
         property = MARSHAL_PREFIX "_discard";
         success = m_cls->hasProperty( m_instance, property );
      }

      if( !success )
      {
         property = MARSHAL_PREFIX "_default";
         success = m_cls->hasProperty( m_instance, property );
         // todo: something more efficient.
         *ctx->local(2) = FALCON_GC_HANDLE( new String(evtName) );
      }

   }

   if( ! success )
   {
      throw new AccessError( ErrorParam( e_marshall_not_found, __LINE__, SRC )
                     .extra(evtName.size() == 0 ? String(MARSHAL_PREFIX) : (MARSHAL_PREFIX + evtName) ));
   }

   ctx->pushCode(&m_stepPropResolved);
   CodeFrame& cf = ctx->currentCode();

   ctx->pushData(Item(m_cls, m_instance));
   m_cls->op_getProperty( ctx, m_instance, property );

   // went deep?
   if( &cf == &ctx->currentCode() )
   {
      // No? -- jump in now
      m_stepPropResolved.apply( &m_stepPropResolved, ctx );
   }
}


void EventMarshal::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      Function::gcMark(mark);
      m_cls->gcMarkInstance(m_instance, mark);
   }
}


void EventMarshal::PStepPropResolved::apply_( const PStep*, VMContext* ctx )
{
   static PStep* psRet = &Engine::instance()->stdSteps()->m_returnFrame;

   // prepare to return the frame.
   ctx->resetCode(psRet);

   Class* cls = 0;
   void* data = 0;
   ctx->topData().forceClassInst( cls, data );

   int base = 0;
   if( ! ctx->local(2)->isNil() )
   {
      base = 1;
      Item temp = *ctx->local(2);
      ctx->pushData(temp);
   }

   if( ctx->local(1)->isTrue() )
   {
      fassert( ctx->local(0)->isArray());
      // our callable is already below.
      ItemArray* arr = ctx->local(0)->asArray();
      for( uint32 i = 0; i < arr->length(); ++i ) {
         ctx->pushData( arr->at(i) );
      }
      cls->op_call(ctx, arr->length()+base, data );
   }
   else {
      Item temp = *ctx->local(0);
      ctx->pushData( temp );
      cls->op_call(ctx, 1+base, data );
   }
}

}

/* end of eventmarshall.cpp */
