/*
   FALCON - The Falcon Programming Language.
   FILE: classstringstream.cpp

   Falcon core module -- String stream interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/stringstream.cpp"

#include <falcon/classes/classstream.h>
#include <falcon/classes/classstringstream.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/stringstream.h>
#include <falcon/stdhandlers.h>

#include <falcon/vm.h>

namespace Falcon {

//=============================================================
//

ClassStringStream::ClassStringStream():
      ClassStream("StringStream"),
      FALCON_INIT_PROPERTY(pipeMode),
      FALCON_INIT_PROPERTY(content),

      FALCON_INIT_METHOD(closeToString)
{
   static Class* ssc = Engine::handlers()->streamClass();
   addParent(ssc);
}

ClassStringStream::~ClassStringStream()
{}

void* ClassStringStream::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassStringStream::op_init( VMContext* ctx, void*, int pcount ) const
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

   StringStream* ss = new StringStream( (int32) count );
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, ss));
   return true;
}

FALCON_DEFINE_PROPERTY_GET_P(ClassStringStream, pipeMode)
{
   StringStream* ss = static_cast<StringStream*>(instance);
   value.setBoolean(ss->isPipeMode());
}

FALCON_DEFINE_PROPERTY_SET_P(ClassStringStream, pipeMode)
{
   StringStream* ss = static_cast<StringStream*>(instance);
   ss->setPipeMode(value.isTrue());
}

FALCON_DEFINE_PROPERTY_GET_P(ClassStringStream, content)
{
   StringStream* ss = static_cast<StringStream*>(instance);
   String* ret = new String;
   ss->getString(*ret);
   ret->manipulator(ret->manipulator()->membufManipulator());
   value = FALCON_GC_HANDLE( ret );
}

FALCON_DEFINE_PROPERTY_SET_P0(ClassStringStream, content)
{
   throw readOnlyError();
}

FALCON_DEFINE_METHOD_P1(ClassStringStream, closeToString )
{
   StringStream* ss = static_cast<StringStream*>(ctx->self().asInst());

   String* ret = ss->closeToString();
   ret->manipulator(ret->manipulator()->membufManipulator());
   ctx->returnFrame( FALCON_GC_HANDLE( ret ) );
}

}

/* end of classstringstream.cpp */
