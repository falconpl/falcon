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
#include <falcon/errors/codeerror.h>

namespace Falcon {
namespace Ext {


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
   return new ItemArray;
}


bool ClassWaiter::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   static Class* shared = Engine::instance()->sharedClass();

   ItemArray* items = static_cast<ItemArray*>(instance);
   Item* params = ctx->opcodeParams(pcount);

   items->append((int64)1);
   for( int32 i = 0; i < pcount; ++ i ) {
      Item* param = params + i;
      if( ! param->isUser() || ! param->asClass()->isDerivedFrom(shared) )
      {
         throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_runtime)
                        .extra("Shared...") );
      }

      items->append(*param);
   }

   return false;
}

void ClassWaiter::op_in( VMContext* ctx, void* instance ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   ctx->opcodeParam(1).setBoolean(self->find(ctx->topData()) >= 0 );
   ctx->popCode();
}

void ClassWaiter::dispose( void* instance ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   delete self;
}

void* ClassWaiter::clone( void* instance ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   return self->clone();
}

void ClassWaiter::gcMarkInstance( void* instance, uint32 mark ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   self->gcMark(mark);
}

bool ClassWaiter::gcCheckInstance( void* instance, uint32 mark ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   return self->currentMark() >= mark ;
}

void ClassWaiter::describe( void* instance, String& target, int, int ) const
{
   ItemArray* self = static_cast<ItemArray*>(instance);
   target = "Waiter(on ";
   target.N(self->length()-1).A(" shared)");
}


void ClassWaiter::store( VMContext*, DataWriter*, void* ) const
{
   // nothing to do
}

void ClassWaiter::restore( VMContext* ctx, DataReader*) const
{
   ItemArray* ir = new ItemArray;
   ctx->pushData( Item(this,ir) );
}

void ClassWaiter::flatten( VMContext*, ItemArray& arr, void* inst ) const
{
   ItemArray* self = static_cast<ItemArray*>(inst);
   arr.merge(*self);
}

void ClassWaiter::unflatten( VMContext*, ItemArray& arr, void* inst ) const
{
   ItemArray* self = static_cast<ItemArray*>(inst);
   self->merge( arr );
}



//========================================================================

FALCON_DEFINE_PROPERTY_GET_P( ClassWaiter, len )
{
   value =(int64) static_cast<ItemArray*>(instance)->length();
}


FALCON_DEFINE_PROPERTY_SET_P0( ClassWaiter, len )
{
   throw readOnlyError();
}


//========================================================================


static void internal_wait( VMContext* ctx, numeric to )
{
   static Class* clsShared = Engine::instance()->sharedClass();
   static PStep* step = &Engine::instance()->stdSteps()->m_waitComplete;
   static PStep* stepInvoke = &Engine::instance()->stdSteps()->m_reinvoke;

   // return if we have pending signals -- we'll be called back.
   if( ctx->releaseAcquired())
   {
      ctx->pushCode(stepInvoke);
      return;
   }

   ItemArray* array = static_cast<ItemArray*>(ctx->self().asInst());


   ctx->initWait();
   uint32 start = (uint32) array->at(0).asInteger();
   uint32 len = array->length();
   array->at(0).setInteger(start+1 >= len? 1: start+1);

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

   for( uint32 i = 1; i < start; ++i )
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
      // return the resource.
      ctx->returnFrame(Item(sh->handler(), sh));
   }
   else if( to  == 0 )
   {
      // return nil immediately
      ctx->returnFrame();
   }
   else {
      // try later.
      ctx->pushCode( step );
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

   internal_wait( ctx, timeout );
}

FALCON_DEFINE_METHOD_P1( ClassWaiter, tryWait )
{
   internal_wait( ctx, 0 );
}

FALCON_DEFINE_METHOD_P1( ClassWaiter, add )
{
   static Class* clsShared = Engine::instance()->sharedClass();

   Item* i_added = ctx->param(0);
   if( i_added == 0 || ! i_added->isUser() || ! i_added->asClass()->isDerivedFrom(clsShared) )
   {
      throw paramError(__LINE__, SRC);
   }

   ItemArray* items = static_cast<ItemArray*>(ctx->self().asInst());
   items->append(*i_added);

   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassWaiter, remove )
{
   static Class* clsShared = Engine::instance()->sharedClass();

   Item* i_added = ctx->param(0);
   if( i_added == 0 || ! i_added->isUser() || ! i_added->asClass()->isDerivedFrom(clsShared) )
   {
      throw paramError(__LINE__, SRC);
   }

   ItemArray* items = static_cast<ItemArray*>(ctx->self().asInst());
   length_t pos = items->find(*i_added);
   if( pos < items->length())
   {
      items->remove(pos);
   }

   ctx->returnFrame();
}


}
}

/* end of parallel.cpp */
