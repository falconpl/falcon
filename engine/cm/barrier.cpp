/*
   FALCON - The Falcon Programming Language.
   FILE: barrier.cpp

   Falcon core module -- Barrier shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/barrier.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/barrier.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>

#include <stdio.h>

namespace Falcon {
namespace Ext {

SharedBarrier::SharedBarrier( ContextManager* mgr, const Class* owner, bool isOpen ):
   Shared( mgr, owner, false, isOpen ? 1 : 0 )
{
   m_status = isOpen ? 1 : 0;
}

SharedBarrier::~SharedBarrier()
{}


int32 SharedBarrier::consumeSignal( VMContext*, int32 )
{
   return atomicFetch(m_status);
}

int32 SharedBarrier::lockedConsumeSignal( VMContext*, int32)
{
   return atomicFetch(m_status);
}

void SharedBarrier::open()
{
   if( atomicCAS(m_status, 0, 1) ) {
      signal(1);
   }
}

void SharedBarrier::close()
{
   if( atomicCAS(m_status, 1, 0) ) {
      Shared::consumeSignal(0, 1);
   }
}


//=============================================================
//

ClassBarrier::ClassBarrier():
      ClassShared("Barrier"),
      FALCON_INIT_PROPERTY(isOpen),
      FALCON_INIT_METHOD(open),
      FALCON_INIT_METHOD(close),
      FALCON_INIT_METHOD(wait)
{
   static Class* shared = Engine::instance()->sharedClass();
   addParent(shared);
}

ClassBarrier::~ClassBarrier()
{}

void* ClassBarrier::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassBarrier::op_init( VMContext* ctx, void*, int pcount ) const
{
   bool isOpen;

   if( pcount == 0 )
   {
      isOpen = false;
   }
   else {
      isOpen = ctx->param(0)->isTrue();
   }

   SharedBarrier* sb = new SharedBarrier(&ctx->vm()->contextManager(), this, isOpen);
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sb));
   return false;
}



FALCON_DEFINE_PROPERTY_GET_P( ClassBarrier, isOpen )
{
   SharedBarrier* sc = static_cast<SharedBarrier*>(instance);
   value.setBoolean( sc->consumeSignal(0) > 0 );
}


FALCON_DEFINE_PROPERTY_SET( ClassBarrier, isOpen )( void*, const Item& )
{
   throw readOnlyError();
}


FALCON_DEFINE_METHOD_P1( ClassBarrier, open )
{
   SharedBarrier* barrier = static_cast<SharedBarrier*>(ctx->self().asInst());
   barrier->open();
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassBarrier, close )
{
   SharedBarrier* barrier = static_cast<SharedBarrier*>(ctx->self().asInst());
   barrier->close();
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P( ClassBarrier, wait )
{
   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}

}
}

/* end of semaphore.cpp */
