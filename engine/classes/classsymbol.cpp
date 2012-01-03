/*
   FALCON - The Falcon Programming Language.
   FILE: classsymbol.cpp

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classsymbol.cpp"

#include <falcon/setup.h>
#include <falcon/classes/classsymbol.h>

#include <falcon/symbol.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>

namespace Falcon {

ClassSymbol::ClassSymbol():
   Class("$Symbol")
{}

ClassSymbol::~ClassSymbol()
{}

void ClassSymbol::dispose( void* ) const
{
   /*Symbol* sym = static_cast<Symbol*>(instance);
   if( sym->type() == Symbol::e_st_closed )
   {
      delete
   }
   */
}


void ClassSymbol::describe( void* instance, String& target, int, int ) const
{
   Symbol* sym = static_cast<Symbol*>( instance );
   target = sym->name();
}

void* ClassSymbol::clone( void* instance ) const
{
   return instance;
}

void ClassSymbol::op_create( VMContext* ctx, int32 pcount ) const
{
   // we DO NOT CREATE SYMBOLS DYNAMICALLY. 
   // This symbols (variables) can be created only via compiler.
   // To create dynamic symbols we have DynSymbols (named "Symbol" to script).
   Class::op_create( ctx, pcount );
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
   cb("value", *sym->defaultValue() );
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
      // maybe to be garbaged.
      ctx->stackResult(2, sym->name().clone()->garbage() );
   }
   else if( prop == "value" )
   {
      ctx->stackResult(2, *sym->value(ctx) );
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
      *sym->value(ctx) = ctx->opcodeParam(3);
      ctx->stackResult(3, *sym->value(ctx) );
      ctx->topData().copied();
   }
   else {
      Class::op_setProperty(ctx, instance, prop);
   }
}


void ClassSymbol::op_eval( VMContext* ctx, void* instance ) const
{
   Symbol* sym = static_cast<Symbol*>( instance );   
   ctx->topData() = *sym->value(ctx);
}
   
}

/* end of classsymbol.cpp */
