/*
   FALCON - The Falcon Programming Language.
   FILE: mutex.cpp

   Falcon core module -- Classic reentrant mutex
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/mutex.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/mutex.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>
#include <falcon/errors/accesserror.h>

#include <stdio.h>

namespace Falcon {
namespace Ext {

SharedMutex::SharedMutex( ContextManager* mgr, const Class* owner ):
   Shared( mgr, owner, true, 1 ) // initially open
{
   m_count = 0;
}

SharedMutex::~SharedMutex()
{}


void SharedMutex::addLock()
{
   atomicInc(m_count);
}


int32 SharedMutex::removeLock()
{
   return atomicDec(m_count);
}



int32 SharedMutex::consumeSignal( VMContext* ctx, int32 )
{
   if( Shared::consumeSignal(ctx, 1) )
   {
      atomicSet(m_count,1);
      return 1;
   }

   return 0;
}

int32 SharedMutex::lockedConsumeSignal(VMContext* ctx, int32)
{
   if( Shared::lockedConsumeSignal(ctx, 1) )
   {
      atomicSet(m_count,1);
      return 1;
   }

   return 0;
}



//=============================================================
//

ClassMutex::ClassMutex():
      ClassShared("Mutex"),
      FALCON_INIT_METHOD(lock),
      FALCON_INIT_METHOD(locked),
      FALCON_INIT_METHOD(tryLock),
      FALCON_INIT_METHOD(unlock)
{
   static Class* shared = Engine::instance()->sharedClass();
   addParent(shared);
}

ClassMutex::~ClassMutex()
{}

void* ClassMutex::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassMutex::op_init( VMContext* ctx, void*, int pcount ) const
{
   SharedMutex* sb = new SharedMutex(&ctx->vm()->contextManager(), this );
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sb));
   return false;
}


//=================================================================================
//


FALCON_DEFINE_METHOD_P1( ClassMutex, lock )
{
   static const PStep& stepWaitSuccess = Engine::instance()->stdSteps()->m_waitSuccess;
   static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;

   SharedMutex* mtx = static_cast<SharedMutex*>(ctx->self().asInst());

   if( ctx->acquired() == mtx )
   {
      // adds a reentrant lock.
      mtx->addLock();
      ctx->returnFrame();
   }
   else
   {
      // first of all check that we're clear to go with pending events.
      ctx->releaseAcquired();
      if( ctx->events() != 0 )
      {
         // i'll be called again, but next time events should be 0.
         ctx->pushCode( &stepInvoke );
         return;
      }

      ctx->initWait();
      ctx->addWait(mtx);
      Shared* sh = ctx->engageWait( -1 );

      if( sh != 0 )
      {
         ctx->returnFrame();
      }
      else {
         // we got to wait.
         ctx->pushCode( &stepWaitSuccess );
      }
   }
}


FALCON_DEFINE_METHOD_P1( ClassMutex, locked )
{
   static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;

   Item* i_param = ctx->param(0);
   if( i_param == 0 )
   {
      throw paramError(__LINE__, SRC );
   }

   SharedMutex* mtx = static_cast<SharedMutex*>(ctx->self().asInst());

   if( ctx->acquired() == mtx )
   {
      // adds a reentrant lock.
      mtx->addLock();
   }
   else
   {
      // first of all check that we're clear to go with pending events.
      ctx->releaseAcquired();
      if( ctx->events() != 0 )
      {
         // i'll be called again, but next time events should be 0.
         ctx->pushCode( &stepInvoke );
         return;
      }

      // prepare the locked call.
      ctx->pushCode( &static_cast<ClassMutex*>(methodOf())->m_stepUnlockAndReturn );

      Class* cls =0;
      void* inst=0;
      i_param->forceClassInst( cls, inst );
      ctx->pushData( *i_param );
      cls->op_call(ctx,0, inst);

      // When we're back, either now or later, we'll have the item locked.
      ctx->initWait();
      ctx->addWait(mtx);
      ctx->engageWait( -1 );
   }
}


void ClassMutex::PStepUnlockAndReturn::apply_( const PStep*, VMContext* ctx )
{
   ctx->releaseAcquired();
   ctx->returnFrame(ctx->topData());
}



FALCON_DEFINE_METHOD_P1( ClassMutex, tryLock )
{
   static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;

   SharedMutex* mtx = static_cast<SharedMutex*>(ctx->self().asInst());

   if( ctx->acquired() == mtx )
   {
      // adds a reentrant lock.
      mtx->addLock();
      ctx->returnFrame( Item().setBoolean(true) );
   }
   else
   {
      // first of all check that we're clear to go with pending events.
      if( ctx->releaseAcquired() )
      {
         // i'll be called again, but next time events should be 0.
         ctx->pushCode( &stepInvoke );
         return;
      }

      ctx->initWait();
      ctx->addWait(mtx);
      Shared* sh = ctx->engageWait( 0 );

      ctx->returnFrame( Item().setBoolean( sh == 0 ) );
   }
}


FALCON_DEFINE_METHOD_P1( ClassMutex, unlock )
{
   SharedMutex* mtx = static_cast<SharedMutex*>(ctx->self().asInst());

   if( ctx->acquired() == mtx )
   {
      // removes a reentrant lock
      if( mtx->removeLock() == 0 )
      {
         // last lock? -- we're off
         ctx->releaseAcquired();
      }
      ctx->returnFrame();
   }
   else
   {
      throw FALCON_SIGN_XERROR( AccessError, e_ctx_ownership,
               .extra("Mutex not owned by this context") );
   }
}

}
}

/* end of semaphore.cpp */
