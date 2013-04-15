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
#include <falcon/stdhandlers.h>


#include <falcon/vm.h>

namespace Falcon {
namespace Ext {


/*#
  @method signal Semaphore
  @brief Signals the semaphore
  @optparam count Count of signals to be sent to the semaphore.

   The parameter @b count must be greater or equal to 1.
 */
FALCON_DECLARE_FUNCTION( post, "count:[N]" );

void Function_post::invoke(VMContext* ctx, int32 pCount)
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
   sm->signal( (int) count );
   ctx->returnFrame();
}


//=============================================================
//

SharedSemaphore::SharedSemaphore( ContextManager* mgr, const Class* owner, int32 initCount ):
   Shared( mgr, owner, false, initCount )
{
}

SharedSemaphore::~SharedSemaphore()
{}


//=============================================================
//

ClassSemaphore::ClassSemaphore():
      ClassShared("Semaphore")
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);

   addMethod( new Function_post );
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

   SharedSemaphore* sm = new SharedSemaphore( &ctx->vm()->contextManager(), this, (int) count);
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sm));
   return true;
}


}
}

/* end of semaphore.cpp */
