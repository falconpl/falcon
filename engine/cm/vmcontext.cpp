/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/vmcontext.cpp"

#include <falcon/cm/vmcontext.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
namespace Ext {


ClassVMContext::ClassVMContext():
   ClassUser("VMContext"),
   
   FALCON_INIT_PROPERTY( id ),
   FALCON_INIT_PROPERTY( processId ),
   FALCON_INIT_PROPERTY( callDepth ),
   FALCON_INIT_PROPERTY( dataDepth ),
   FALCON_INIT_PROPERTY( codeDepth ),
   FALCON_INIT_PROPERTY( selfItem ),
   FALCON_INIT_PROPERTY( status ),
   FALCON_INIT_METHOD( caller )
{
}

ClassVMContext::~ClassVMContext()
{}


void* ClassVMContext::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassVMContext::dispose( void* instance ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   ctx->decref();
}

void* ClassVMContext::clone( void* instance ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   ctx->incref();
   return ctx;
}


void ClassVMContext::gcMarkInstance( void* instance, uint32 mark ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   // not a really good idea, but...
   ctx->gcStartMark(mark);
   ctx->gcPerformMark();
}

bool ClassVMContext::gcCheckInstance( void* instance, uint32 mark ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   return ctx->currentMark() >= mark;
}

bool ClassVMContext::op_init( VMContext* ctx, void*, int pcount ) const
{
   ctx->stackResult(pcount, FALCON_GC_STORE(this, ctx));
   return true;
}

void ClassVMContext::op_toString( VMContext* ctx, void* instance ) const
{
   VMContext* ctx1 = static_cast<VMContext*>(instance);
   String& res = * new String();
   if( ctx1 != ctx ) {
      res+="Foreign context";
   }
   else {
      res.append("VMContext {");
      res.N(ctx1->id()).A(":").N(ctx1->process()->id()).A("}");
      ctx->topData().setUser(FALCON_GC_HANDLE(&res));
   }
}

//====================================================
// Properties.
//
   
FALCON_DEFINE_PROPERTY_SET( ClassVMContext, id )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("id") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, id )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->id());
}

FALCON_DEFINE_PROPERTY_SET( ClassVMContext, status )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("status") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, status )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->getStatus());
}


FALCON_DEFINE_PROPERTY_SET( ClassVMContext, processId )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("processId") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, processId )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->process()->id());
}

FALCON_DEFINE_PROPERTY_SET( ClassVMContext, callDepth )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("callDepth") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, callDepth )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->callDepth());
}


FALCON_DEFINE_PROPERTY_SET( ClassVMContext, dataDepth )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("dataDepth") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, dataDepth )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->dataSize());
}

FALCON_DEFINE_PROPERTY_SET( ClassVMContext, codeDepth )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("codeDepth") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, codeDepth )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->codeDepth());
}

FALCON_DEFINE_PROPERTY_SET( ClassVMContext, selfItem )(void*, const Item& )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("selfItem") );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassVMContext, selfItem )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.assign(ctx->currentFrame().m_self);
}

//========================================================================
// Methods
//

FALCON_DEFINE_METHOD( ClassVMContext, caller )(VMContext* ctx, int )
{
   Item* i_depth = ctx->param(0);
   int32 depth = 1;
   if( i_depth != 0 ) {
      if( ! i_depth->isOrdinal() ) {
         throw paramError(__LINE__, SRC);
      }
      depth = i_depth->forceInteger();
   }

   // we're not interested in this call
   depth++;
   // we can't use a foreign context, sorry.
   VMContext* tgt = ctx;
   if( depth >= tgt->callDepth() ) {
      throw new ParamError(
               ErrorParam( e_param_range, __LINE__, SRC )
                  .extra("excessive depth")
               );
   }

   //ugly but works
   CallFrame* cf = &(&tgt->currentFrame())[-depth];
   if( cf->m_bMethodic ) {
      Item s = cf->m_self;
      s.methodize(cf->m_function);
      ctx->returnFrame(s);
   }
   else {
      ctx->returnFrame(cf->m_function);
   }
}

}
}

/* end of vmcontext.cpp */
