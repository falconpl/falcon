/*
   FALCON - The Falcon Programming Language.
   FILE: classdynsymbol.cpp

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 22:08:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classdynsymbol.cpp"

#include <falcon/setup.h>
#include <falcon/classes/classdynsymbol.h>

#include <falcon/symbol.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/dynsymbol.h>
#include <falcon/errors/paramerror.h>
#include <falcon/collector.h>


namespace Falcon {

ClassDynSymbol::ClassDynSymbol():
   Class("Symbol")
{}

ClassDynSymbol::~ClassDynSymbol()
{}

void ClassDynSymbol::describe( void* instance, String& target, int, int ) const
{
   DynSymbol* sym = static_cast<DynSymbol*>(instance);   
   target = sym->name();
}

void ClassDynSymbol::dispose( void* instance ) const
{   
   DynSymbol* sym = static_cast<DynSymbol*>(instance);   
   delete sym;
}

void* ClassDynSymbol::clone( void* instance ) const
{
   return instance;
}

void ClassDynSymbol::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   Item* item = ctx->opcodeParams(pcount);
   if( pcount > 0 || item->isString() )
   {
      DynSymbol* sym = new DynSymbol( *item->asString() );
      ctx->stackResult( pcount+1, Item( FALCON_GC_STORE(coll, this, sym) ) );
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra("S"));
   }
   
}

void ClassDynSymbol::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("name", false);
   cb("value", true);
}

void ClassDynSymbol::enumeratePV( void* instance, PVEnumerator& cb ) const
{
   static Class* strClass = Engine::instance()->stringClass();

   DynSymbol* sym = static_cast<DynSymbol*>( instance );
   Item temp( strClass, sym );
   cb("name", temp );
   //cb("value", *sym->defaultValue() );
}

bool ClassDynSymbol::hasProperty( void*, const String& prop ) const
{
   return
      prop == "name"
      || prop == "value";
}

void ClassDynSymbol::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   DynSymbol* sym = static_cast<DynSymbol*>( instance );

   if( prop == "name" )
   {
      ctx->stackResult(2, sym->name().clone()->garbage() );
   }
   else if( prop == "value" )
   {
      ctx->stackResult(2, *ctx->getDynSymbolValue(sym) );
      ctx->topData().copied();
   }
   else {
      Class::op_getProperty(ctx, instance, prop);
   }
}


void ClassDynSymbol::op_setProperty( VMContext* ctx, void* instance, const String& prop) const
{
   DynSymbol* sym = static_cast<DynSymbol*>( instance );

   if( prop == "name" )
   {
      throw ropError( prop, __LINE__, SRC );
   }
   else if( prop == "value" )
   {
      *ctx->getDynSymbolValue(sym) = ctx->opcodeParam(3);
      ctx->popData(2);
      ctx->topData().copied();
   }
   else {
      Class::op_setProperty(ctx, instance, prop);
   }
}


void ClassDynSymbol::op_eval( VMContext* ctx, void* instance ) const
{
   DynSymbol* sym = static_cast<DynSymbol*>( instance );   
   ctx->topData() = *ctx->getDynSymbolValue(sym);
}
   
}

/* end of classdynsymbol.cpp */
