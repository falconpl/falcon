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


namespace Falcon {

ClassSymbol::ClassSymbol():
   Class("Symbol")
{}

ClassSymbol::~ClassSymbol()
{}

void ClassSymbol::describe( void* instance, String& target, int, int ) const
{
   Symbol* sym = static_cast<Symbol*>(instance);   
   target = sym->name();
}

void ClassSymbol::dispose( void* instance ) const
{   
   Symbol* sym = static_cast<Symbol*>(instance);   
   delete sym;
}

void* ClassSymbol::clone( void* instance ) const
{
   return instance;
}

void ClassSymbol::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   Item* item = ctx->opcodeParams(pcount);
   if( pcount > 0 || item->isString() )
   {
      Symbol* sym = new Symbol( *item->asString() );
      ctx->stackResult( pcount+1, Item( FALCON_GC_STORE(coll, this, sym) ) );
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra("S"));
   }
   
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
      ctx->stackResult(1, sym->name().clone()->garbage() );
   }
   else if( prop == "value" )
   {
      ctx->stackResult(1, *sym->getValue(ctx) );
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
      sym->setValue(ctx, ctx->opcodeParam(3));
      ctx->popData(2);
      ctx->topData().copied();
   }
   else {
      Class::op_setProperty(ctx, instance, prop);
   }
}


void ClassSymbol::op_eval( VMContext* ctx, void* instance ) const
{
   Symbol* sym = static_cast<Symbol*>( instance );   
   ctx->topData() = *sym->getValue(ctx);
}
   
}

/* end of classsymbol.cpp */
