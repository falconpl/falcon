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
#include <falcon/symboltable.h>
#include <falcon/callframe.h>
#include <falcon/itemarray.h>

#include "falcon/module.h"

namespace Falcon {

Symbol::Symbol()
{
   // leave all unconfigured.
}

Symbol::Symbol( const String& name, int line ):
   m_name(name),
   m_declaredAt(line),         
   m_remote(0),   
   m_type( Symbol::e_st_dynamic ),
   m_bConstant(false)  
{
   m_host.module = 0;
   m_defvalue.asItem = 0;
   
   m_getValue = Symbol::getValue_dyns;
   m_setValue = Symbol::setValue_dyns;

}
   
Symbol::Symbol( const String& name, SymbolTable* host, uint32 id, int line ):
   m_name(name),
   m_declaredAt(line),         
   m_remote(0),   
   m_type( Symbol::e_st_local ),
   m_bConstant(false)  
{
   m_host.symtab = host;
   m_defvalue.asId = id;
   
   m_getValue = Symbol::getValue_local;
   m_setValue = Symbol::setValue_local;
}
   
Symbol::Symbol( const String& name, Module* host, Item* value, int line ):
   m_name(name),
   m_declaredAt(line),         
   m_remote(0),   
   m_type( Symbol::e_st_global ),
   m_bConstant(false)  
{
   m_host.module = host;
   m_defvalue.asItem = value;
   
   m_getValue = Symbol::getValue_global;
   m_setValue = Symbol::setValue_global;
}
   
   
Symbol* Symbol::ExternSymbol( const String& name, Module* mod, int line )
{
   Symbol* sym = new Symbol;
   sym->m_name = name;
   sym->m_declaredAt = line;         
   sym->m_remote = 0;
   sym->m_type = e_st_extern;
   sym->m_bConstant = false;  
   
   sym->m_host.module = mod;
   sym->m_defvalue.asItem = 0;
   
   sym->m_getValue = Symbol::getValue_extern;
   sym->m_setValue = Symbol::setValue_extern;
   return sym;
}
   

Symbol* Symbol::ClosedSymbol( const String& name, SymbolTable* host, uint32 id, int line )
{
   Symbol* sym = new Symbol;
   sym->m_name = name;
   sym->m_declaredAt = line;         
   sym->m_remote = 0;
   sym->m_type = e_st_extern;
   sym->m_bConstant = false;  
   
   sym->m_host.symtab = host;
   sym->m_defvalue.asId = id;
   
   sym->m_getValue = Symbol::getValue_closed;
   sym->m_setValue = Symbol::setValue_closed;
   return sym;
}
   

Symbol::Symbol( const Symbol& other ):
   m_name(other.m_name),
   m_declaredAt(other.m_declaredAt),         
   m_defvalue(other.m_defvalue),
   m_host(other.m_host),
   m_remote(other.m_remote),
   m_setValue(other.m_setValue),
   m_getValue(other.m_getValue),
   m_type( other.m_type ),
   m_bConstant(other.m_bConstant)
{
}


Symbol::~Symbol()
{
}

void Symbol::gcMark( uint32 mark )
{
   if( m_name.currentMark() < mark )
   {
      m_name.gcMark( mark );
      switch( m_type )
      {
         case e_st_global:
            m_host.module->gcMark( mark );
            break;
            
         case e_st_closed:
         case e_st_local:
            m_host.symtab->gcMark( mark );
            break;
            
         case e_st_extern:
            m_host.module->gcMark( mark );
            if( m_remote != 0 ) m_remote->gcMark( mark );
            break;
            
            // else, there's nothing to mark;
            // we trust that both dynamic symbols have values somewhere safe.
         case e_st_dynamic:  break;
      }
   }
}


void Symbol::resolveExtern( Module* newHost, Item* value )
{
   fassert2( m_type == e_st_extern, "Ought to be an extern symbol." );
   m_remote = newHost;
   m_defvalue.asItem = value;
   
   m_getValue = Symbol::getValue_global;
   m_setValue = Symbol::setValue_global;
}

void Symbol::promoteExtern( Item* value )
{
   fassert2( m_type == e_st_extern, "Ought to be an extern symbol." );
   m_defvalue.asItem = value;
   // change also the type
   m_type = e_st_global;
   
   m_getValue = Symbol::getValue_global;
   m_setValue = Symbol::setValue_global;
}


Item* Symbol::getValue_global( const Symbol* sym, VMContext* )
{
   return sym->m_defvalue.asItem;
}

Item* Symbol::getValue_local( const Symbol* sym, VMContext* ctx )
{
   return &ctx->localVar(sym->m_defvalue.asId);
}

Item* Symbol::getValue_closed( const Symbol* sym, VMContext* ctx )
{
   return & (*ctx->currentFrame().m_closedData)[sym->m_defvalue.asId];
}

Item* Symbol::getValue_extern( const Symbol* sym, VMContext* ctx )
{
   // operate as a dynamic value
   return ctx->getDynSymbolValue( sym );
}

Item* Symbol::getValue_dyns( const Symbol* sym, VMContext* ctx )
{
   return ctx->getDynSymbolValue( sym );
}


void Symbol::setValue_global( const Symbol* sym, VMContext*, const Item& value )
{
   sym->m_defvalue.asItem->assign(value);
}

void Symbol::setValue_local( const Symbol* sym, VMContext* ctx, const Item& value)
{
   ctx->localVar(sym->m_defvalue.asId).assign(value);
}

void Symbol::setValue_closed( const Symbol* sym, VMContext* ctx, const Item& value )
{
   (*ctx->currentFrame().m_closedData)[sym->m_defvalue.asId].assign(value);
}

void Symbol::setValue_extern( const Symbol* sym, VMContext* ctx, const Item& value )
{
   // operate as a dynamic value
   ctx->setDynSymbolValue( sym, value );
}


void Symbol::setValue_dyns( const Symbol* sym, VMContext* ctx, const Item& value )
{
   ctx->setDynSymbolValue( sym, value );
}

}

/* end of symbol.cpp */
