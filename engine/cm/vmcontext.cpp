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
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
namespace Ext {

//====================================================
// Properties.
//

static void get_id( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->id());
}


static void get_status( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->getStatus());
}


static void get_processId( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->process()->id());
}


static void get_callDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->callDepth());
}


static void get_dataDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->dataSize());
}


static void get_codeDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.setInteger(ctx->codeDepth());
}


static void get_selfItem( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   value.assign(ctx->currentFrame().m_self);
}

//========================================================================
// Methods
//

/*#
 @method caller VMContext
 @brief Returns the item (function or method) that is calling the current function.
 @optparam depth If specified, return the nth parameter up to @a VMContext.codeDepth

 If @b depth is not specified, it defaults to 1. Using 0 returns the same entity as
 obtained by the @b fself keyword.
 */
FALCON_DECLARE_FUNCTION( caller, "depth:[N]" );
void Function_caller::invoke(VMContext* ctx, int32 )
{
   Item* i_depth = ctx->param(0);
   int32 depth = 1;
   if( i_depth != 0 ) {
      if( ! i_depth->isOrdinal() ) {
         throw paramError(__LINE__, SRC);
      }
      depth = (int32) i_depth->forceInteger();
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



//====================================================


ClassVMContext::ClassVMContext():
   Class("VMContext")
{   
   addProperty( "id", &get_id );
   addProperty( "processId", &get_processId );
   addProperty( "callDepth", &get_callDepth );
   addProperty( "dataDepth", &get_dataDepth );
   addProperty( "codeDepth", &get_codeDepth );
   addProperty( "selfItem", &get_selfItem );
   addProperty( "status", &get_status );
   
   addMethod( new Function_caller );
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

}
}

/* end of vmcontext.cpp */
