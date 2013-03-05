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
#include <falcon/stdhandlers.h>
#include <falcon/vm.h>
#include <falcon/function.h>

#include <stdio.h>

namespace Falcon {
namespace Ext {

/*#
  @property isOpen Barrier
  @brief Checks if the barrier is open in this moment.
 */
static void get_isOpen( const Class*, const String&, void* instance, Item& value )
{
   SharedBarrier* sc = static_cast<SharedBarrier*>(instance);
   value.setBoolean( sc->consumeSignal(0) > 0 );
}

/*#
  @method open Barrier
  @brief opens the barrier.

  Opening an already open barrier has no effect; the first @a Barrier.close
  call will close the barrier, no matter how many open are issued.
 */
FALCON_DECLARE_FUNCTION( open, "" );

/*#
  @method close Barrier
  @brief Closes the barrier.

  Closing the barrier will cause any agent waiting on the barrier
  from that moment on to be blocked.
 */
FALCON_DECLARE_FUNCTION( close, "" );
void Function_open::invoke(VMContext* ctx, int32 )
{
   SharedBarrier* barrier = static_cast<SharedBarrier*>(ctx->self().asInst());
   barrier->open();
   ctx->returnFrame();
}


void Function_close::invoke(VMContext* ctx, int32 )
{
   SharedBarrier* barrier = static_cast<SharedBarrier*>(ctx->self().asInst());
   barrier->close();
   ctx->returnFrame();
}


//=======================================================================
//


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
      ClassShared("Barrier")
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);

   addProperty("isOpen", &get_isOpen);
   addMethod( new Function_open );
   addMethod( new Function_close );
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

}
}

/* end of semaphore.cpp */
