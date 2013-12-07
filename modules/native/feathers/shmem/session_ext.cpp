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
#include "errors.h"

#include <falcon/trace.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/stderrors.h>
#include <falcon/symbol.h>
#include <falcon/itemdict.h>

namespace Falcon {

namespace {

FALCON_DECLARE_FUNCTION(init, "id:S,mode:[N]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   TRACE1("Session.init() with %d parameters", ctx->paramCount() );

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

   TRACE1("Session.init(\"%s\", %d)", id.c_ize(), mode );

   ctx->returnFrame(ctx->self());
}


static void internal_add_remove( Function* func, VMContext* ctx, int32 pCount, bool bAdd, int minCount = 1 )
{
   if( pCount < minCount )
   {
      throw func->paramError(__LINE__);
   }

   Session* session = ctx->tself<Session*>();

   for(int32 i = 0; i < pCount; i++ )
   {
      Item* item = ctx->param(i);
      fassert( item != 0 );
      if( item->isString() )
      {
         const String& symName = *item->asString();
         Symbol* sym = Engine::getSymbol(symName);
         if( bAdd ) {
            session->addSymbol(sym);
         }
         else {
            session->removeSymbol(sym);
         }
         sym->decref();
      }
      else if( item->isSymbol() )
      {
         Symbol* sym = item->asSymbol();
         if( bAdd ) {
            session->addSymbol(sym);
         }
         else {
            session->removeSymbol(sym);
         }
      }
      else
      {
         throw func->paramError(String("Parameter ").N(i+1).A(" must be a symbol or a string"));
      }
   }
}


FALCON_DECLARE_FUNCTION(add, "symbol:S|Symbol,...")
FALCON_DEFINE_FUNCTION_P(add)
{
   TRACE1("Session.add() with %d parameters", ctx->paramCount() );
   internal_add_remove( this, ctx, pCount, true );
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(remove, "symbol:S|Symbol,...")
FALCON_DEFINE_FUNCTION_P(remove)
{
   TRACE1("Session.remove() with %d parameters", ctx->paramCount() );
   internal_add_remove( this, ctx, pCount, false );
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(open, "apply:[B]")
FALCON_DEFINE_FUNCTION_P(open)
{
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;

   Session* session = ctx->tself<Session*>();
   bool bApply = pCount > 0 ? ctx->param(0)->isTrue() : true;
   TRACE1("Session.open(%s)", bApply ? "true" : "false" );

#ifndef NDEBUG
   String id;
   session->getID(id);
   TRACE1("Session(%s).open(%s)", id.c_ize(), (bApply ? "true" : "false") );
#endif


   session->open();
   ctx->pushCode( retStep );
   session->load(ctx, bApply);
   // don't return the frame
}


FALCON_DECLARE_FUNCTION(create, "symbol:[S|Symbol],..." )
FALCON_DEFINE_FUNCTION_P(create)
{
   TRACE1("Session.create() with %d parameters", ctx->paramCount() );
   Session* session = ctx->tself<Session*>();

#ifndef NDEBUG
   String id;
   session->getID(id);
   TRACE1("Session(%s).create() with %d parameters", id.c_ize(), ctx->paramCount() );
#endif

   internal_add_remove( this, ctx, pCount, true, 0 );

   session->create();
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(start, "symbol:[S|Symbol],...")
FALCON_DEFINE_FUNCTION_P(start)
{
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;

   Session* session = ctx->tself<Session*>();
   TRACE1("Session.start() with %d parameters", ctx->paramCount() );

   session->open();
   if( ! session->checkLoad() )
   {
      internal_add_remove( this, ctx, pCount, true, 0 );
      session->begin();
   }

   ctx->pushCode( retStep );
   session->load(ctx, true);
   // don't return the frame
}


FALCON_DECLARE_FUNCTION(close, "")
FALCON_DEFINE_FUNCTION_P1(close)
{
   MESSAGE("Session.close()" );
   Session* session = ctx->tself<Session*>();
   session->close();

   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(apply, "")
FALCON_DEFINE_FUNCTION_P1(apply)
{
   MESSAGE("Session.apply()" );
   Session* session = ctx->tself<Session*>();
   session->apply(ctx);
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(save, "")
FALCON_DEFINE_FUNCTION_P1(save)
{
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;

   MESSAGE("Session.save()" );
   Session* session = ctx->tself<Session*>();
   ctx->pushCode( retStep );
   session->record(ctx);
   session->save(ctx);
   // don't return the frame
}


FALCON_DECLARE_FUNCTION(get, "symbol:S|Symbol,dflt:[X]")
FALCON_DEFINE_FUNCTION_P1(get)
{
   MESSAGE("Session.get()" );
   Symbol* sym;

   Item* i_symbol = ctx->param(0);
   Item* i_dflt = ctx->param(1);

   if( i_symbol == 0 )
   {
      throw paramError(__LINE__, SRC);
   }
   else if( i_symbol->isString() )
   {
      sym = Engine::getSymbol(*i_symbol->asString());
   }
   else if( i_symbol->isSymbol() )
   {
      sym = i_symbol->asSymbol();
   }
   else {
      throw paramError(__LINE__, SRC);
   }

   TRACE("Session.get(\"%s\")", sym->name().c_ize() );
   Item value;
   Session* session = ctx->tself<Session*>();
   bool found = session->get(sym, value);

   if ( i_symbol->isString() )
   {
      sym->decref();
   }

   if ( found )
   {
      ctx->returnFrame( value );
   }
   else {
      if( i_dflt != 0 )
      {
         ctx->returnFrame(*i_dflt);
      }
      else {
         throw FALCON_SIGN_XERROR( AccessError, e_dict_acc, .extra("in session") );
      }
   }
}


FALCON_DECLARE_FUNCTION(set, "symbol:S|Symbol,value:X")
FALCON_DEFINE_FUNCTION_P1(set)
{
   MESSAGE("get.set()" );
   Symbol* sym;

   Item* i_symbol = ctx->param(0);
   Item* i_value = ctx->param(1);

   if( i_symbol == 0 || i_value == 0 )
   {
      throw paramError(__LINE__, SRC);
   }
   else if( i_symbol->isString() )
   {
      sym = Engine::getSymbol(*i_symbol->asString());
   }
   else if( i_symbol->isSymbol() )
   {
      sym = i_symbol->asSymbol();
   }
   else {
      throw paramError(__LINE__, SRC);
   }

   TRACE("Session.set(\"%s\", ...)", sym->name().c_ize() );
   Item value;
   Session* session = ctx->tself<Session*>();
   session->addSymbol(sym, *i_value);

   ctx->returnFrame();
}



FALCON_DECLARE_FUNCTION(getAll, "")
FALCON_DEFINE_FUNCTION_P1(getAll)
{
   MESSAGE("getAll()");
   Session* session = ctx->tself<Session*>();

   ItemDict* dict = new ItemDict;

   class Rator: public Session::Enumerator
   {
   public:
      Rator( ItemDict* dict ): m_dict(dict) {}
      virtual ~Rator() {}
      virtual void operator()(Symbol* sym, Item& value)
      {
         m_dict->insert(FALCON_GC_HANDLE(new String(sym->name()) ), value);
      }

   private:
      ItemDict* m_dict;
   };

   Rator rator(dict);

   session->enumerate(rator);
   ctx->returnFrame(FALCON_GC_HANDLE(dict));
}

}

//=============================================================================
// Session class handler
//=============================================================================

ClassSession::ClassSession():
         Class("Session")
{
   setConstuctor(new FALCON_FUNCTION_NAME(init) );
   addMethod( new FALCON_FUNCTION_NAME(add) );
   addMethod( new FALCON_FUNCTION_NAME(remove) );

   addMethod( new FALCON_FUNCTION_NAME(open) );
   addMethod( new FALCON_FUNCTION_NAME(create) );
   addMethod( new FALCON_FUNCTION_NAME(start) );
   addMethod( new FALCON_FUNCTION_NAME(close) );
   addMethod( new FALCON_FUNCTION_NAME(apply) );
   addMethod( new FALCON_FUNCTION_NAME(save) );

   addMethod( new FALCON_FUNCTION_NAME(get) );
   addMethod( new FALCON_FUNCTION_NAME(set) );
   addMethod( new FALCON_FUNCTION_NAME(getAll) );

   addConstant("OM_FILE", static_cast<int64>(Session::e_om_file) );
   addConstant("OM_SHMEM", static_cast<int64>(Session::e_om_shmem) );
   addConstant("OM_SHMEM_BU", static_cast<int64>(Session::e_om_shmem_bu) );
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
