/*
   FALCON - The Falcon Programming Language.
   FILE: semaphore.cpp

   Falcon core module -- Semaphore shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/semaphore.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/semaphore.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>

#include <falcon/vm.h>

namespace Falcon {
namespace Ext {

SharedSemaphore::SharedSemaphore( ContextManager* mgr, const Class* owner, int32 initCount ):
   Shared( mgr, owner, false, initCount )
{
}

SharedSemaphore::~SharedSemaphore()
{}


//=============================================================
//

ClassSemaphore::ClassSemaphore():
      ClassShared("Semaphore"),
      FALCON_INIT_METHOD(post),
      FALCON_INIT_METHOD(wait)
{
   static Class* shared = Engine::instance()->sharedClass();
   addParent(shared);
}

ClassSemaphore::~ClassSemaphore()
{}

void* ClassSemaphore::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassSemaphore::op_init( VMContext* ctx, void*, int pcount ) const
{
   int64 count;

   if( pcount == 0 )
   {
      count = 0;
   }
   else {
      Item* i_count = ctx->opcodeParams(pcount);
      if( ! i_count->isOrdinal() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("N") );
      }
      count = i_count->forceInteger();
      if( count < 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra(">=0") );
      }
   }

   SharedSemaphore* sm = new SharedSemaphore( &ctx->vm()->contextManager(), this, count);
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sm));
   return true;
}


FALCON_DEFINE_METHOD_P( ClassSemaphore, post )
{
   int64 count;

   if( pCount == 0 )
   {
      count = 1;
   }
   else {
      Item* i_count = ctx->param(0);
      if( ! i_count->isOrdinal() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("[N]") );
      }
      count = i_count->forceInteger();
      if( count <= 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra(">0") );
      }
   }

   SharedSemaphore* sm = static_cast<SharedSemaphore*>(ctx->self().asInst());
   // The semaphore can't be the acquired resource, as it's not acquireable.
   sm->signal( count );
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P( ClassSemaphore, wait )
{
   static const PStep& stepWaitSuccess = Engine::instance()->stdSteps()->m_waitSuccess;
   static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;

   //===============================================
   //
   int64 timeout = -1;
   if( pCount >= 1 )
   {
      Item* i_timeout = ctx->param(0);
      if (!i_timeout->isOrdinal())
      {
         throw paramError(__LINE__, SRC);
      }

      timeout = i_timeout->forceInteger();
   }

   // first of all check that we're clear to go with pending events.
   if( ctx->releaseAcquired() )
   {
      // i'll be called again, but next time events should be 0.
      ctx->pushCode(&stepInvoke);
      return;
   }

   SharedSemaphore* sm = static_cast<SharedSemaphore*>(ctx->self().asInst());
   ctx->initWait();
   ctx->addWait(sm);
   Shared* shared = ctx->engageWait( timeout );

   if( shared != 0 )
   {
      ctx->returnFrame( Item().setBoolean(true) );
   }
   else {
      // we got to wait.
      ctx->pushCode( &stepWaitSuccess );
   }
}

}
}

/* end of semaphore.cpp */
