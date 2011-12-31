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

namespace Falcon {

ClassSymbol::ClassSymbol():
   Class("Symbol")
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

void* ClassSymbol::clone( void* instance ) const
{
   return instance;
}
   
void ClassSymbol::op_create( VMContext* ctx, int32 pcount ) const
{
   // TODO
   ctx->stackResult(pcount+1, Item());
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
   static Class* strClass = Engine::instance()->stringClass();
   
   Symbol* sym = static_cast<Symbol*>( instance );
   
   if( prop == "name" )
   {
      ctx->stackResult(2, Item( strClass, sym ) );
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

}

/* end of classsymbol.cpp */
