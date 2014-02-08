/*
   FALCON - The Falcon Programming Language.
   FILE: classsymbol.cpp

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 22:08:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classSymbol.cpp"

#include <falcon/setup.h>
#include <falcon/classes/classsymbol.h>
#include <falcon/stdhandlers.h>

#include <falcon/symbol.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/symbol.h>
#include <falcon/stderrors.h>
#include <falcon/collector.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <falcon/itemarray.h>
#include <falcon/synclasses_id.h>


namespace Falcon {

ClassSymbol::ClassSymbol():
   Class("Symbol", FLC_CLASS_ID_SYMBOL )
{
}

ClassSymbol::~ClassSymbol()
{}

void ClassSymbol::describe( void* instance, String& target, int, int ) const
{
   Symbol* sym = static_cast<Symbol*>(instance);
   target = "$";
   target += sym->name();
}

void ClassSymbol::dispose( void* instance ) const
{   
   Symbol* sym = static_cast<Symbol*>(instance);
   Engine::releaseSymbol(sym);
}

void* ClassSymbol::clone( void* instance ) const
{
   Symbol* sym = static_cast<Symbol*>(instance);
   Engine::refSymbol(sym);
   return sym;
}

void* ClassSymbol::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

bool ClassSymbol::op_init( VMContext* ctx, void*, int32 pcount ) const
{   
   Item* item = ctx->opcodeParams(pcount);
   if( pcount > 0 || item->isString() )
   {
      String& name = *item->asString();
      Symbol* sym = Engine::getSymbol( name );
      FALCON_GC_STORE( this, sym );
      ctx->opcodeParam(pcount).setUser( this, sym );
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra("S"));
   }
   
   return false;
}

void ClassSymbol::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("name");
   cb("value");
}

void ClassSymbol::enumeratePV( void* instance, PVEnumerator& cb ) const
{
   static Class* strClass = Engine::handlers()->stringClass();

   Symbol* sym = static_cast<Symbol*>( instance );
   Item temp( strClass, sym );
   cb("name", temp );
   //cb("value", *sym->defaultValue() );
}

bool ClassSymbol::hasProperty( void*, const String& prop ) const
{
   return
      prop == "name"
      || prop == "value";
}

void ClassSymbol::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Symbol* sym = static_cast<Symbol*>( instance );

   if( prop == "name" )
   {
      ctx->stackResult(1, FALCON_GC_HANDLE(sym->name().clone()) );
   }
   else if( prop == "value" )
   {
      ctx->stackResult(1, *ctx->resolveSymbol(sym, false) );
   }
   else {
      Class::op_getProperty(ctx, instance, prop);
   }
}


void ClassSymbol::op_setProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Symbol* sym = static_cast<Symbol*>( instance );

   if( prop == "name" )
   {
      FALCON_RESIGN_ROPROP_ERROR(prop, ctx);
   }
   else if( prop == "value" )
   {
      ctx->resolveSymbol( sym, true )->copyFromLocal(ctx->opcodeParam(3));
   }
   else {
      Class::op_setProperty(ctx, instance, prop);
   }
}

void ClassSymbol::op_call( VMContext* ctx, int pcount, void* instance ) const
{
   Symbol* sym = static_cast<Symbol*>( instance );
   if( pcount > 0 )
   {
      ctx->opcodeParam(pcount) = ctx->opcodeParam(pcount-1);
      ctx->popData(pcount);
      *ctx->resolveSymbol(sym, true) = ctx->topData();
   }
   else {
      ctx->popData(pcount);
      ctx->topData() = *ctx->resolveSymbol(sym, false);
   }
}


void ClassSymbol::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Symbol* symbol = static_cast<Symbol*>(instance);
   stream->write( symbol->name() );
}


void ClassSymbol::restore( VMContext* ctx, DataReader* stream ) const
{
   String name;
   
   stream->read( name );
   
   Symbol* sym = Engine::getSymbol( name );
   ctx->pushData( Item( this, sym ) );
}

}

/* end of classsymbol.cpp */
