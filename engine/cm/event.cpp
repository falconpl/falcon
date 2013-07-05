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
#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
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


int32 SharedEvent::consumeSignal( VMContext* ctx, int32 )
{
   return Shared::consumeSignal(ctx);
}

int32 SharedEvent::lockedConsumeSignal( VMContext* ctx, int32)
{
   return Shared::lockedConsumeSignal(ctx, 1);
}

void SharedEvent::set()
{
   lockSignals();
   if( ! lockedSignalCount() ) {
      lockedSignal(1);
   }
   unlockSignals();
}

//=============================================================
//

/*#
  @method set Event
  @brief Sets the event

 */
FALCON_DECLARE_FUNCTION( set, "" );
void Function_set::invoke(VMContext* ctx, int32 )
{
   SharedEvent* evt = static_cast<SharedEvent*>(ctx->self().asInst());
   evt->set();
   ctx->returnFrame();
}

//=============================================================
//

ClassEvent::ClassEvent():
      ClassShared("Event")      
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);
   addMethod( new Function_set );
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
   return true;
}

}
}

/* end of semaphore.cpp */
