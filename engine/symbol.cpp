/*
   FALCON - The Falcon Programming Language.
   FILE: sybmol.cpp

   Syntactic tree item definitions -- expression elements -- symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/symbol.h>
#include <falcon/stream.h>
#include <falcon/vm.h>
#include <falcon/callframe.h>
#include <falcon/function.h>
#include <falcon/module.h>
#include <falcon/itemarray.h>
#include <falcon/fassert.h>
#include <falcon/modspace.h>

#include <falcon/module.h>
#include <falcon/closure.h>

namespace Falcon {

Symbol::Symbol():
         m_counter(1)
{
   // leave all unconfigured.
}

Symbol::Symbol( const String& name, bool isGlobal ):
   m_name(name),
   m_isGlobal( isGlobal ),
   m_counter(1)
{
}
   

   
Symbol::Symbol( const Symbol& other ):
   m_name(other.m_name),
   m_isGlobal( other.m_isGlobal ),
   m_counter(1)
{
}


Symbol::~Symbol()
{
}


void Symbol::incref()
{
   Engine::refSymbol( this );
}

void Symbol::decref()
{
   Engine::releaseSymbol( this );
}

Item* Symbol::resolve( VMContext* ctx, bool forAssign ) const
{
   TRACE1( "Symbol::resolve -- resolving %s%s", m_name.c_ize(), m_isGlobal ? " (global)": "")
   CallFrame* cf = &ctx->currentFrame();
   Function* func = cf->m_function;

   if( ! m_isGlobal )
   {
      Variable* var = func->variables().find( m_name );
      if( var != 0 ) {
         switch( var->type() ) {
         case Variable::e_nt_closed:
            if( cf->m_closure == 0 ) return 0;
            return cf->m_closure->get(m_name);

         case Variable::e_nt_local:
            return ctx->local(var->id());

         case Variable::e_nt_param:
            return ctx->param(var->id());

         default:
            fassert2( false, "Shouldn't have this in a function" );
            break;
         }
      }
      else if( forAssign ) {
         ctx->pushData(Item());
         return &ctx->topData();
      }
    }


   // didn't find it locally, try globally
   Module* mod = func->module();
   if( mod != 0 ) {
      // findGlobal will find also externally resolved variables.
      Item* global = mod->getGlobalValue( m_name );
      if( global != 0 ) {
         return global;
      }
      else if( forAssign ) {
         Variable* var = mod->addGlobal( m_name, Item(), false );
         global = mod->getGlobalValue( var->id() );
         return global;
      }
   }

   // try as non-imported extern
   if( ! forAssign )
   {
      // if the module space is the same as the vm modspace,
      // mod->findGlobal has already searched for it
      Item* item = ctx->vm()->modSpace()->findExportedValue( m_name );
      if( item != 0 ) {
         return item;
      }
   }

   // no luck
   return 0;
}

}

/* end of symbol.cpp */
