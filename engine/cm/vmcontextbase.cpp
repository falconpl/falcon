/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontextbase.cpp

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

#include <falcon/cm/vmcontextbase.h>

#include <falcon/vm.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/processor.h>
#include <falcon/stderrors.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
namespace Ext {

//====================================================
// Properties.
//

static void get_id( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->id());
}


static void get_status( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->getStatus());
}


static void get_processId( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->process()->id());
}


static void get_callDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->callDepth());
}


static void get_dataDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->dataSize());
}


static void get_codeDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->codeDepth());
}

static void get_dynsDepth( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.setInteger(ctx->dynsDepth());
}


static void get_selfItem( const Class*, const String&, void* instance, Item& value )
{
   VMContext* ctx;
   if( instance == 0 )
   {
      ctx = Processor::currentProcessor()->currentContext();
   }
   else {
      ctx = static_cast<VMContext*>(instance);
   }

   value.copyInterlocked(ctx->currentFrame().m_self);
}


//====================================================


ClassVMContextBase::ClassVMContextBase():
   Class("#VMContext")
{   
   init();
}

ClassVMContextBase::ClassVMContextBase( const String& name ):
   Class(name)
{
   init();
}


ClassVMContextBase::~ClassVMContextBase()
{}


void ClassVMContextBase::init()
{
   addProperty( "id", &get_id );
   addProperty( "processId", &get_processId );
   addProperty( "callDepth", &get_callDepth );
   addProperty( "dataDepth", &get_dataDepth );
   addProperty( "codeDepth", &get_codeDepth );
   addProperty( "dynsDepth", &get_dynsDepth );
   addProperty( "selfItem", &get_selfItem );
   addProperty( "status", &get_status );
}

void* ClassVMContextBase::createInstance() const
{
   return 0;
}

void ClassVMContextBase::dispose( void* instance ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   if( ctx != 0 )
   {
      ctx->decref();
   }
}

void* ClassVMContextBase::clone( void* instance ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   if( ctx != 0 )
   {
      ctx->incref();
   }
   return ctx;
}


void ClassVMContextBase::gcMarkInstance( void* instance, uint32 mark ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   if( ctx != 0 )
   {
      ctx->gcStartMark(mark);
      ctx->gcPerformMark();
   }
}


bool ClassVMContextBase::gcCheckInstance( void* instance, uint32 mark ) const
{
   VMContext* ctx = static_cast<VMContext*>(instance);
   if( ctx != 0 )
   {
      return ctx->currentMark() >= mark;
   }

   return true;
}


void ClassVMContextBase::op_toString( VMContext* ctx, void* instance ) const
{
   VMContext* ctx1 = static_cast<VMContext*>(instance);
   if( ctx1 == 0 )
   {
      ctx1 = Processor::currentProcessor()->currentContext();
   }

   String &res = *(new String);

   res.append("VMContext {");
   res.N(ctx1->id()).A(":").N(ctx1->process()->id()).A("}");
   ctx->topData().setUser(FALCON_GC_HANDLE(&res));
}

}
}

/* end of vmcontext.cpp */
