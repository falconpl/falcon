/*
   FALCON - The Falcon Programming Language.
   FILE: classshared.cpp

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Nov 2012 12:52:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classshared.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/shared.h>
#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>
#include <falcon/stdsteps.h>
#include <falcon/function.h>

namespace Falcon
{
namespace CShared {
FALCON_DECLARE_FUNCTION( tryWait, "" );
void Function_tryWait::invoke( VMContext* ctx, int32 pCount )
{
   ClassShared::genericClassTryWait(methodOf(), ctx, pCount);
}

FALCON_DECLARE_FUNCTION( wait, "timeout:[N]" );
void Function_wait::invoke( VMContext* ctx, int32 pCount )
{
   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}
}


static void reg( Class* shared )
{
   shared->addMethod( new CShared::Function_tryWait );
   shared->addMethod( new CShared::Function_wait );
}


ClassShared::ClassShared( const String& name ):
         Class(name)
{
   reg( this );
}

ClassShared::ClassShared( const String& name, int64 type ):
         Class(name, (int32) type)
{
   reg( this );
}


ClassShared::ClassShared():
         Class("Shared")
{
   reg( this );
}


ClassShared::~ClassShared()
{
}


void ClassShared::dispose( void* self ) const
{
   Shared* sh = static_cast<Shared*>(self);
   sh->decref();
}

void* ClassShared::clone( void* source ) const
{
   Shared* sh = static_cast<Shared*>(source);
   return sh->clone();
}


void* ClassShared::createInstance() const
{
   // this is a virtual class.
   return 0;
}

void ClassShared::describe( void* instance, String& target, int, int ) const
{
   target.A("<").A(name()).A("* ").N((int64) instance).A(">");
}

void ClassShared::gcMarkInstance( void* self, uint32 mark ) const
{
   Shared* sh = static_cast<Shared*>(self);
   sh->gcMark( mark );
}

bool ClassShared::gcCheckInstance( void* self, uint32 mark ) const
{
   Shared* sh = static_cast<Shared*>(self);
   return sh->currentMark() >= mark;
}


void ClassShared::genericClassTryWait( const Class* , VMContext* ctx, int32 )
{
   // first of all check that we're clear to go with pending events.
   if( ctx->releaseAcquired() )
   {
      // i'll be called again, but next time events should be 0.
      static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;
      ctx->pushCode( &stepInvoke );
      return;
   }

   Shared* shared = static_cast<Shared*>(ctx->self().asInst());
   bool result = shared->consumeSignal(ctx, 1) > 0;
   ctx->returnFrame( Item().setBoolean(result) );
}


void ClassShared::genericClassWait( const Class* childClass, VMContext* ctx, int32 pCount )
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
         throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("[N]") );
      }

      timeout = i_timeout->forceInteger();
   }

   if( timeout == 0 )
   {
      // use a simpler approach
      genericClassTryWait(childClass, ctx, pCount );
      return;
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
   Shared* engaged = ctx->engageWait( timeout );

   if( engaged == shared )
   {
      ctx->returnFrame( Item().setBoolean(true) );
   }
   else {
      // we got to wait.
      if( timeout == 0 )
      {
         ctx->returnFrame(Item().setBoolean(false));
      }
      else {
         shared->onWaiterWaiting(ctx, timeout);
         ctx->pushCode( &stepWaitSuccess );
      }
   }
}


}

/* end of classshared.cpp */
