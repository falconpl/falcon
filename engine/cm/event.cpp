/*
   FALCON - The Falcon Programming Language.
   FILE: event.cpp

   Falcon core module -- Event shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/event.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/event.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>

#include <stdio.h>

namespace Falcon {
namespace Ext {

SharedEvent::SharedEvent( ContextManager* mgr, const Class* owner, bool isSet ):
   Shared( mgr, owner, false, isSet? 1:0 )
{

}

SharedEvent::~SharedEvent()
{}


int32 SharedEvent::consumeSignal( int32 )
{
   lockSignals();
   if( atomicCAS(m_status, 1, 0) )
   {
      Shared::lockedConsumeSignal(1);
      unlockSignals();
      return 1;
   }
   unlockSignals();
   return 0;
}

int32 SharedEvent::lockedConsumeSignal(int32)
{
   if( atomicCAS(m_status, 1, 0) )
   {
      Shared::lockedConsumeSignal(1);
      return 1;
   }
   return 0;
}

void SharedEvent::set()
{
   if( atomicCAS(m_status, 0, 1) ) {
      signal(1);
   }
}


//=============================================================
//

ClassEvent::ClassEvent():
      ClassShared("Event"),
      FALCON_INIT_METHOD(set),
      FALCON_INIT_METHOD(wait)
{
   static Class* shared = Engine::instance()->sharedClass();
   addParent(shared);
}

ClassEvent::~ClassEvent()
{}

void* ClassEvent::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassEvent::op_init( VMContext* ctx, void*, int pcount ) const
{
   bool isSet;

   if( pcount == 0 )
   {
      isSet = false;
   }
   else {
      isSet = ctx->param(0)->isTrue();
   }

   SharedEvent* sb = new SharedEvent(&ctx->vm()->contextManager(), this, isSet);
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sb));
   return false;
}


FALCON_DEFINE_METHOD_P1( ClassEvent, set )
{
   SharedEvent* evt = static_cast<SharedEvent*>(ctx->self().asInst());
   evt->set();
   ctx->returnFrame();
}



FALCON_DEFINE_METHOD_P( ClassEvent, wait )
{
   static const PStep& stepWaitSuccess = Engine::instance()->stdSteps()->m_waitSuccess;

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
      static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;
      ctx->pushCode( &stepInvoke );
      return;
   }

   Shared* shared = static_cast<Shared*>(ctx->self().asInst());
   ctx->initWait();
   ctx->addWait(shared);
   shared = ctx->engageWait( timeout );

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
