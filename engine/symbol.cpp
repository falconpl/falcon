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
   m_type( Symbol::e_st_dynamic ),
   m_bConstant(false)
{
   m_id = 0;
   m_getValue = Symbol::getValue_dyns;
   m_setValue = Symbol::setValue_dyns;
}
   
Symbol::Symbol( const String& name, Symbol::type_t t, uint32 id, int line ):
   m_name(name),
   m_declaredAt(line),         
   m_type( t ),
   m_bConstant(false)  
{
   m_id = id;
   
   switch(t)
   {
      case e_st_local:
         m_getValue = Symbol::getValue_local;
         m_setValue = Symbol::setValue_local;
         break;
         
      case e_st_global:
         m_getValue = Symbol::getValue_global;
         m_setValue = Symbol::setValue_global;
         break;
         
      case e_st_closed:
         m_getValue = Symbol::getValue_closed;
         m_setValue = Symbol::setValue_closed;
         break;
         
      case e_st_extern:
         m_getValue = Symbol::getValue_extern_unpromoted;
         m_setValue = Symbol::setValue_extern;
         break;
         
      case e_st_dynamic:
         m_getValue = Symbol::getValue_dyns;
         m_setValue = Symbol::setValue_dyns;
         break;
   }
   
}
   

Symbol::Symbol( const Symbol& other ):
   m_name(other.m_name),
   m_declaredAt(other.m_declaredAt),         
   m_setValue(other.m_setValue),
   m_getValue(other.m_getValue),
   m_type( other.m_type ),
   m_bConstant(other.m_bConstant)
{
   m_defValue = other.m_defValue;
   m_other = other.m_other;
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
         case e_st_extern:
            if( m_other != 0 ) {
               m_other->gcMark( mark );
            }
            break;
            
         // Dynamic symbols have their value 
         case e_st_dynamic:  
            break;
         
         // local symbols may have default values to be marked (e.g. params).
         default:
            m_defValue.gcMark(mark);
      }
   }
}

const Item* Symbol::getValue_global( const Symbol* sym, VMContext* )
{
   return &sym->m_defValue;
}

const Item* Symbol::getValue_local( const Symbol* sym, VMContext* ctx )
{
   return &ctx->localVar(sym->m_id);
}

const Item* Symbol::getValue_closed( const Symbol* sym, VMContext* ctx )
{
   return & (*ctx->currentFrame().m_closedData)[sym->m_id];
}

const Item* Symbol::getValue_extern( const Symbol* sym, VMContext* )
{
   fassert( sym->m_other != 0 );
   return &sym->m_other->m_defValue;
}

const Item* Symbol::getValue_extern_unpromoted( const Symbol*, VMContext* )
{
   return 0;
}

const Item* Symbol::getValue_dyns( const Symbol* sym, VMContext* ctx )
{
   return ctx->getDynSymbolValue( sym );
}


void Symbol::setValue_global( const Symbol* sym, VMContext*, const Item& value )
{
   const_cast<Symbol*>(sym)->m_defValue.assign(value);
}

void Symbol::setValue_local( const Symbol* sym, VMContext* ctx, const Item& value)
{
   ctx->localVar(sym->m_id).assign(value);
}

void Symbol::setValue_closed( const Symbol* sym, VMContext* ctx, const Item& value )
{
   (*ctx->currentFrame().m_closedData)[sym->m_id].assign(value);
}

void Symbol::setValue_extern( const Symbol* sym, VMContext*, const Item& value )
{
   fassert( sym->m_other != 0 );
   sym->m_other->m_defValue = value;
}


void Symbol::setValue_dyns( const Symbol* sym, VMContext* ctx, const Item& value )
{
   ctx->setDynSymbolValue( sym, value );
}

}

/* end of symbol.cpp */
