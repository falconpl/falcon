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

#include <falcon/symbol.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/symbol.h>
#include <falcon/errors/paramerror.h>
#include <falcon/collector.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include "falcon/itemarray.h"


namespace Falcon {

ClassSymbol::ClassSymbol():
   Class("Symbol")
{}

ClassSymbol::~ClassSymbol()
{}

void ClassSymbol::describe( void* instance, String& target, int, int ) const
{
   Symbol* sym = static_cast<Symbol*>(instance);
   target = "&";
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
      bool isFirst = false;
      Symbol* sym = Engine::getSymbol(name, false, isFirst );
      if( isFirst ) {
         FALCON_GC_STORE( this, sym );
      }
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
   cb("name", false);
   cb("value", true);
}

void ClassSymbol::enumeratePV( void* instance, PVEnumerator& cb ) const
{
   static Class* strClass = Engine::instance()->stringClass();

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
      ctx->topData().copied();
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
      throw ropError( prop, __LINE__, SRC );
   }
   else if( prop == "value" )
   {
      ctx->resolveSymbol( sym, true )->assign(ctx->opcodeParam(3));
   }
   else {
      Class::op_setProperty(ctx, instance, prop);
   }
}

void ClassSymbol::op_call( VMContext* ctx, int pcount, void* instance ) const
{
   Symbol* sym = static_cast<Symbol*>( instance );
   ctx->popData(pcount);
   ctx->topData() = *ctx->resolveSymbol(sym, false);
}


void ClassSymbol::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Symbol* symbol = static_cast<Symbol*>(instance);
   stream->write( symbol->name() );
   stream->write( (char) (symbol->isGlobal()?1:0) );
}


void ClassSymbol::restore( VMContext* ctx, DataReader* stream ) const
{
   String name;
   char type;
   
   stream->read( name );
   stream->read( type );
   
   Symbol* sym = Engine::getSymbol( name, type != 0 ? true : false );
   ctx->pushData( FALCON_GC_STORE( this, sym ) );
}

}

/* end of classsymbol.cpp */
