/*
   FALCON - The Falcon Programming Language.
   FILE: session_ext.cpp

   Falcon script interface for Inter-process persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/modules/native/feathers/shmem/session_ext.cpp"

#include "session_ext.h"
#include "session.h"

#include <falcon/function.h>
#include <falcon/vmcontext.h>

namespace Falcon {

namespace {

FALCON_DECLARE_FUNCTION(init, "id:S,mode:[N]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   Item* i_id = ctx->param(0);
   Item* i_mode = ctx->param(1);

   if( i_id == 0 || ! i_id->isString()
       || ( i_mode != 0 && ! i_mode->isOrdinal()) )
   {
      throw paramError(__LINE__, SRC);
   }

   int mode = (int) (i_mode != 0 ? i_mode->forceInteger() : (int) Session::e_om_shmem );
   if( mode != static_cast<int>(Session::e_om_shmem)
       && mode != static_cast<int>(Session::e_om_shmem_bu)
       && mode != static_cast<int>(Session::e_om_file)
   )
   {
      throw paramError( "Invalid mode", __LINE__, SRC);
   }

   Session* session = ctx->tself<Session*>();

   const String& id = *i_id->asString();
   session->setID(id);
   session->setOpenMode(static_cast<Session::t_openmode>(mode));

   ctx->returnFrame(ctx->self());
}


FALCON_DECLARE_FUNCTION(add, "symbol:S|Symbol,...")
FALCON_DEFINE_FUNCTION_P1(add)
{

   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(open, "apply:[B]")
FALCON_DEFINE_FUNCTION_P1(open)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(create, "")
FALCON_DEFINE_FUNCTION_P1(create)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(close, "")
FALCON_DEFINE_FUNCTION_P1(close)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(apply, "")
FALCON_DEFINE_FUNCTION_P1(apply)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(save, "")
FALCON_DEFINE_FUNCTION_P1(save)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(get, "symbol:S|Symbol")
FALCON_DEFINE_FUNCTION_P1(get)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(set, "symbol:S|Symbol,value:X")
FALCON_DEFINE_FUNCTION_P1(set)
{
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(remove, "symbol:S|Symbol")
FALCON_DEFINE_FUNCTION_P1(remove)
{
   ctx->returnFrame();
}

FALCON_DECLARE_FUNCTION(getAll, "")
FALCON_DEFINE_FUNCTION_P1(getAll)
{
   ctx->returnFrame();
}

}

//=============================================================================
// Session class handler
//=============================================================================

ClassSession::ClassSession():
         Class("Session")
{
   addMethod( new FALCON_FUNCTION_NAME(init) );
   addMethod( new FALCON_FUNCTION_NAME(add) );
   addMethod( new FALCON_FUNCTION_NAME(open) );
   addMethod( new FALCON_FUNCTION_NAME(create) );
   addMethod( new FALCON_FUNCTION_NAME(close) );
   addMethod( new FALCON_FUNCTION_NAME(apply) );
   addMethod( new FALCON_FUNCTION_NAME(save) );

   addMethod( new FALCON_FUNCTION_NAME(get) );
   addMethod( new FALCON_FUNCTION_NAME(set) );
   addMethod( new FALCON_FUNCTION_NAME(remove) );
   addMethod( new FALCON_FUNCTION_NAME(getAll) );
}


ClassSession::~ClassSession()
{
}


int64 ClassSession::occupiedMemory( void* inst ) const
{
   Session* s = static_cast<Session*>(inst);

   return s->occupiedMemory();
}


void* ClassSession::createInstance() const
{
   return new Session;
}


void ClassSession::dispose( void* instance ) const
{
   Session* s = static_cast<Session*>(instance);
   delete s;
}


void* ClassSession::clone( void* ) const
{
   return 0;
}

void ClassSession::describe( void* instance, String& target, int depth, int maxlen ) const
{
   //TODO
   Class::describe(instance, target, depth, maxlen);
}

}

/* end of session_ext.cpp */
