/*
   FALCON - The Falcon Programming Language.
   FILE: selector.cpp

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Nov 2012 12:52:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classselector.cpp"

#include <falcon/selector.h>
#include <falcon/cm/selector.h>
#include <falcon/stream.h>
#include <falcon/errors/paramerror.h>
#include <falcon/shared.h>
#include <falcon/vm.h>

#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdsteps.h>

namespace Falcon {
namespace Ext {

ClassSelector::ClassSelector():
         ClassShared("Selector"),
         FALCON_INIT_METHOD(add),
         FALCON_INIT_METHOD(update),
         FALCON_INIT_METHOD(addRead),
         FALCON_INIT_METHOD(addWrite),
         FALCON_INIT_METHOD(addErr),

         FALCON_INIT_METHOD(getRead),
         FALCON_INIT_METHOD(getWrite),
         FALCON_INIT_METHOD(getErr),
         FALCON_INIT_METHOD(get),

         FALCON_INIT_METHOD(tryWait),
         FALCON_INIT_METHOD(wait)
{
   static Class* shared = Engine::handlers()->sharedClass();
   addParent( shared );
}


ClassSelector::~ClassSelector()
{
}


void* ClassSelector::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassSelector::op_init( VMContext* ctx, void*, int pcount ) const
{
   Item* i_mode = ctx->opcodeParams(pcount);

   bool fair = i_mode == 0 ? false : i_mode->isTrue();
   Selector* sel = new Selector( &ctx->vm()->contextManager(), this, fair );
   ctx->stackResult(pcount+1, FALCON_GC_HANDLE(sel) );

   return true;
}


static void internal_selector_add( VMContext* ctx, int32, bool bAdd )
{
   static Class* streamClass = Engine::handlers()->streamClass();
   Item* i_stream = ctx->param(0);
   Item* i_mode = ctx->param(1);

   Class* cls = 0;
   void* data = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data) || ! cls->isDerivedFrom(streamClass)
       || (i_mode != 0 && ! i_mode->isOrdinal() )
       )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "stream:Stream, mode:[N]") );
   }

   int32 mode;
   if( i_mode == 0 )
   {
      mode = Selector::mode_err | Selector::mode_write | Selector::mode_read;
   }
   else
   {
      mode = (int32)i_mode->forceInteger();
      if( mode < 0 || mode > (Selector::mode_err | Selector::mode_write | Selector::mode_read) )
      {
         throw new ParamError(ErrorParam(e_param_range, __LINE__, SRC).extra( "select mode out of range") );
      }
   }

   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* sc = static_cast<Stream*>( cls->getParentData(streamClass,data) );
   if( bAdd )
   {
      sel->add( sc, mode );
   }
   else {
      sel->update( sc, mode );
   }

   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P( ClassSelector, add )
{
   internal_selector_add(ctx, pCount, true );
}

FALCON_DEFINE_METHOD_P( ClassSelector, update )
{
   internal_selector_add(ctx, pCount, false );
}



static void internal_add_mode( VMContext* ctx, int32, int32 mode )
{
   static Class* streamClass = Engine::handlers()->streamClass();
   Item* i_stream = ctx->param(0);

   Class* cls = 0;
   void* data = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data) || ! cls->isDerivedFrom(streamClass) )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "stream:Stream") );
   }

   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* sc = static_cast<Stream*>( cls->getParentData(streamClass,data) );
   sel->add( sc, mode );

   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P( ClassSelector, addRead )
{
   internal_add_mode( ctx, pCount, Selector::mode_read );
}


FALCON_DEFINE_METHOD_P( ClassSelector, addWrite )
{
   internal_add_mode( ctx, pCount, Selector::mode_write );
}


FALCON_DEFINE_METHOD_P( ClassSelector, addErr )
{
   internal_add_mode( ctx, pCount, Selector::mode_err );
}


FALCON_DEFINE_METHOD_P1( ClassSelector, getRead )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyRead();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}


FALCON_DEFINE_METHOD_P1( ClassSelector, getWrite )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyWrite();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}


FALCON_DEFINE_METHOD_P1( ClassSelector, getErr )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyErr();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}

FALCON_DEFINE_METHOD_P1( ClassSelector, get )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyErr();
   if( stream == 0 ) stream = sel->getNextReadyRead();
   if( stream == 0 ) stream = sel->getNextReadyWrite();

   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}



FALCON_DEFINE_METHOD_P( ClassSelector, tryWait )
{
   ClassShared::genericClassTryWait(methodOf(), ctx, pCount);
}

FALCON_DEFINE_METHOD_P( ClassSelector, wait )
{
   ClassShared::genericClassWait(methodOf(), ctx, pCount);
}


}
}

/* end of selector.cpp */
