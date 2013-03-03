/*
   FALCON - The Falcon Programming Language.
   FILE: waiter.cpp

   Falcon core module -- Wait multiplexer for multiple shared items
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Nov 2012 13:52:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/waiter.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/waiter.h>
#include <falcon/stdhandlers.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/contextgroup.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/stdsteps.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/accesserror.h>

#include <falcon/processor.h>

#include <map>

namespace Falcon {
namespace Ext {


class WaiterData
{
public:
   typedef std::map<Shared*, Item> CallbackMap;

   ItemArray m_waited;
   CallbackMap m_callbacks;

   length_t m_pos;
   volatile VMContext* m_owner;

   WaiterData():
      m_pos(0),
      m_owner(0)
   {}
};


ClassWaiter::ClassWaiter():
   ClassUser("Waiter"),

   FALCON_INIT_PROPERTY( len ),

   FALCON_INIT_METHOD( wait ),
   FALCON_INIT_METHOD( tryWait ),
   FALCON_INIT_METHOD( add ),
   FALCON_INIT_METHOD( remove )
{
}

ClassWaiter::~ClassWaiter()
{}


void* ClassWaiter::createInstance() const
{
   return new WaiterData();
}


bool ClassWaiter::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   static Class* shared = Engine::handlers()->sharedClass();

   WaiterData* wd = static_cast<WaiterData*>(instance);
   ItemArray* items = &wd->m_waited;

   Item* params = ctx->opcodeParams(pcount);

   for( int32 i = 0; i < pcount; ++ i ) {
      Item* param = params + i;
      if( ! param->isUser() || ! param->asClass()->isDerivedFrom(shared) )
      {
         throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_runtime)
                        .extra("Shared...") );
      }

      // no need to lock ATM, we're the only users.
      items->append(*param);
   }
   wd->m_owner = ctx;

   return false;
}

void ClassWaiter::op_in( VMContext* ctx, void* instance ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);
   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   // no need to lock, we're using just the top array.
   ctx->opcodeParam(1).setBoolean(self->m_waited.find(ctx->topData()) >= 0 );
   ctx->popCode();
}

void ClassWaiter::dispose( void* instance ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   delete self;
}

void* ClassWaiter::clone( void* instance ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);

   Processor* proc = Processor::currentProcessor();
   if( proc == 0 || proc->currentContext() != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   WaiterData* nData = new WaiterData;
   nData->m_waited.merge( self->m_waited );
   WaiterData::CallbackMap::iterator iter = nData->m_callbacks.begin();
   WaiterData::CallbackMap::iterator end = nData->m_callbacks.end();

   while( iter != end )
   {
      Shared* sh = iter->first;
      int32 pos = self->m_waited.find(Item(sh->handler(), sh));
      if( pos >= 0 )
      {
         Shared* newShared = static_cast<Shared*>(nData->m_waited.at(pos).asInst());
         nData->m_callbacks[newShared] = iter->second;
      }
      ++iter;
   }
   return nData;
}

void ClassWaiter::gcMarkInstance( void* instance, uint32 mark ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);

   if( self->m_waited.currentMark() != mark )
   {
      self->m_waited.gcMark(mark);
      WaiterData::CallbackMap::iterator iter = self->m_callbacks.begin();
      WaiterData::CallbackMap::iterator end = self->m_callbacks.end();

      while( iter != end )
      {
         iter->second.gcMark(mark);
         ++iter;
      }
   }
}

bool ClassWaiter::gcCheckInstance( void* instance, uint32 mark ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);
   return self->m_waited.currentMark() >= mark;
}

void ClassWaiter::describe( void* instance, String& target, int, int ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);

   Processor* proc = Processor::currentProcessor();
   if( proc == 0 || proc->currentContext() != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   target = "Waiter(on ";
   target.N(self->m_waited.length()).A(" shared)");
}


void ClassWaiter::store( VMContext* ctx, DataWriter*, void* instance ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);

   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }
}

void ClassWaiter::restore( VMContext* ctx, DataReader*) const
{
   WaiterData* ir = new WaiterData;
   ir->m_owner = ctx;
   ctx->pushData( Item(this,ir) );
}

void ClassWaiter::flatten( VMContext* ctx,  ItemArray& arr, void* instance ) const
{
   WaiterData* self = static_cast<WaiterData*>(instance);

   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   arr.append( Item(self->m_waited.handler(), &self->m_waited));
}

void ClassWaiter::unflatten( VMContext*, ItemArray& arr, void* inst ) const
{
   WaiterData* self = static_cast<WaiterData*>(inst);
   fassert( arr[0].asClass() == self->m_waited.handler() );

   self->m_waited.merge( *static_cast<ItemArray*>(arr[0].asInst()) );
}



//========================================================================

FALCON_DEFINE_PROPERTY_GET_P( ClassWaiter, len )
{
   value =(int64) static_cast<WaiterData*>(instance)->m_waited.length();
}


FALCON_DEFINE_PROPERTY_SET_P0( ClassWaiter, len )
{
   throw readOnlyError();
}


//========================================================================


void ClassWaiter::internal_wait( VMContext* ctx, int64 to )
{
   static Class* clsShared = Engine::handlers()->sharedClass();
   static PStep* stepInvoke = &Engine::instance()->stdSteps()->m_reinvoke;

   WaiterData* self = static_cast<WaiterData*>(ctx->self().asInst());
   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   // return if we have pending signals -- we'll be called back.
   if( ctx->releaseAcquired())
   {
      ctx->pushCode(stepInvoke);
      return;
   }

   ItemArray* array = &self->m_waited;

   ctx->initWait();
   uint32 start = self->m_pos;
   uint32 len = array->length();
   self->m_pos++;
   if( self->m_pos >= len ) {
      self->m_pos = 0;
   }

   // roll the loop
   for( uint32 i = start; i < len; ++i )
   {
      Item& param = array->at(i);
      Class* cls = 0;
      void* inst = 0;
      param.asClassInst(cls, inst);

      Shared* sh = static_cast<Shared*>(cls->getParentData( clsShared, inst ));
      ctx->addWait(sh);
   }

   for( uint32 i = 0; i < start; ++i )
   {
      Item& param = array->at(i);
      Class* cls = 0;
      void* inst = 0;
      param.asClassInst(cls, inst);

      Shared* sh = static_cast<Shared*>(cls->getParentData( clsShared, inst ));
      ctx->addWait(sh);
   }

   Shared* sh = ctx->engageWait(to);
   if( sh != 0 )
   {
      returnOrInvoke( ctx, sh );
   }
   else if( to  == 0 )
   {
      // return nil immediately
      ctx->returnFrame();
   }
   else {
      // try later.
      ctx->pushCode( &m_stepAfterWait );
   }
}


void ClassWaiter::returnOrInvoke( VMContext* ctx, Shared* sh )
{
   WaiterData* self = static_cast<WaiterData*>(ctx->self().asInst());

   // return the resource?
   WaiterData::CallbackMap::iterator cbmp = self->m_callbacks.find( sh );
   if( cbmp == self->m_callbacks.end() )
   {
      ctx->returnFrame(Item(sh->handler(), sh));
   }
   else {
      // prepare a step to analyze the function result.
      ctx->pushCode( &m_stepAfterCall );

      // invoke the function.
      ctx->pushData( cbmp->second );
      ctx->pushData( Item(sh->handler(),sh) );
      Class* cls = 0;
      void* inst = 0;
      cbmp->second.asClassInst(cls, inst);
      cls->op_call(ctx,1,inst);
   }
}

void ClassWaiter::PStepAfterCall::apply_(const PStep*, VMContext* ctx )
{
   static PStep* reinvoke = &Engine::instance()->stdSteps()->m_reinvoke;

   Item& result = ctx->topData();
   // get the return value from the invoked function.
   if( result.isOob())
   {
      // the function is asking us to return.
      result.setOob(false);
      ctx->returnFrame(result);
   }
   else {
      ctx->popData();
      // we just need to re-loop
      ctx->resetCode(reinvoke);
   }
}

void ClassWaiter::PStepAfterWait::apply_(const PStep*, VMContext* ctx )
{
   // did we have a success?
   Shared* shared = ctx->getSignaledResouce();
   if( shared != 0 )
   {
      ctx->popCode();
      ClassWaiter* waiter = static_cast<ClassWaiter*>(ctx->self().asClass());
      shared->decref(); // extra ref not needed if we're in garbage system
      waiter->returnOrInvoke( ctx, shared );
   }
   else {
      // we timed out
      ctx->returnFrame();
   }
}



FALCON_DEFINE_METHOD_P1( ClassWaiter, wait )
{
   int64 timeout = -1;
   Item* i_timeout = ctx->param(0);
   if( i_timeout != 0 )
   {
      if( ! i_timeout->isOrdinal() )
      {
         throw paramError(__LINE__, SRC);
      }
      timeout = i_timeout->forceInteger();
   }

   static_cast<ClassWaiter*>(methodOf())->internal_wait( ctx, timeout );
}

FALCON_DEFINE_METHOD_P1( ClassWaiter, tryWait )
{
   static_cast<ClassWaiter*>(methodOf())->internal_wait( ctx, 0 );
}

FALCON_DEFINE_METHOD_P1( ClassWaiter, add )
{
   static Class* clsShared = Engine::handlers()->sharedClass();

   WaiterData* self = static_cast<WaiterData*>(ctx->self().asInst());
   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   Item* i_added = ctx->param(0);
   Item* i_callback = ctx->param(1);

   Class* cls = 0;
   void* inst = 0;

   if( i_added == 0
            || ! i_added->asClassInst(cls, inst)
            || ! cls->isDerivedFrom(clsShared)
     )
   {
      throw paramError(__LINE__, SRC);
   }

   self->m_waited.append(*i_added);
   Shared* sh = static_cast<Shared*>(cls->getParentData(clsShared, inst));
   sh->onWaiterWaiting(ctx);

   if( i_callback != 0 )
   {
      self->m_callbacks[sh] = *i_callback;
   }

   ctx->returnFrame(ctx->self());
}


FALCON_DEFINE_METHOD_P1( ClassWaiter, remove )
{
   static Class* clsShared = Engine::handlers()->sharedClass();

   WaiterData* self = static_cast<WaiterData*>(ctx->self().asInst());
   if( ctx != self->m_owner )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_ctx_ownership);
   }

   Item* i_added = ctx->param(0);
   Class* cls = 0;
   void* inst = 0;
   if( i_added == 0
            || ! i_added->asClassInst(cls, inst)
            || ! cls->isDerivedFrom(clsShared)
            )
   {
      throw paramError(__LINE__, SRC);
   }

   int32 pos = self->m_waited.find(*i_added);
   if( pos < 0)
   {
      self->m_waited.remove(pos);
      Shared* sh = static_cast<Shared*>(cls->getParentData(clsShared, inst));
      self->m_callbacks.erase(sh);
   }

   ctx->returnFrame();
}


}
}

/* end of parallel.cpp */
