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
#include "falcon/closure.h"

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
   m_getVariable = Symbol::getVariable_dyns;
   m_setVariable = Symbol::setVariable_dyns;
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
         m_setVariable = m_getVariable = Symbol::getVariable_local;
         break;
         
      case e_st_global:
         m_setVariable = m_getVariable = Symbol::getVariable_global;
         m_realValue.base(0);
         m_realValue.value(&m_defValue);
         break;
         
      case e_st_closed:
         m_setVariable = m_getVariable = Symbol::getVariable_closed;
         break;
         
      case e_st_extern:
         m_setVariable = m_getVariable = Symbol::getVariable_extern;
         break;
         
      case e_st_dynamic:
         m_getVariable = Symbol::getVariable_dyns;
         m_setVariable = Symbol::setVariable_dyns;
         break;
   }   
}
   

Symbol::Symbol( const Symbol& other ):
   m_name(other.m_name),
   m_declaredAt(other.m_declaredAt),         
   m_getVariable(other.m_getVariable),
   m_type( other.m_type ),
   m_bConstant(other.m_bConstant)
{
   m_defValue = other.m_defValue;
   m_realValue = other.m_realValue;
}


Symbol::~Symbol()
{
}

void Symbol::promoteToGlobal()
{
   fassert2( m_type == e_st_extern, "Ought to be an extern symbol" );
   m_type = e_st_global;
   m_getVariable = &getVariable_global;
   m_realValue.set(0, &m_defValue);
}


void Symbol::resolved(Variable* other)
{
   fassert2( m_type == e_st_extern, "Ought to be an extern symbol" );
   m_type = e_st_global;
   m_getVariable = &getVariable_global;
   m_realValue.makeReference(other);
}

void Symbol::globalWithValue( const Item& value )
{
   m_type = e_st_global;
   m_getVariable = &getVariable_global;
   m_defValue = value;
   m_realValue.set(0, &m_defValue);   
}

void Symbol::gcMark( uint32 mark )
{
   if( m_name.currentMark() < mark )
   {
      m_name.gcMark( mark );
      m_defValue.gcMark(mark);
      if( m_realValue.base() != 0 ) {
         *m_realValue.base() = mark;
      }
      m_realValue.value()->gcMark(mark);      
   }
}

Variable* Symbol::getVariable_global( Symbol* sym, VMContext* )
{
   return &sym->m_realValue;
}

Variable* Symbol::getVariable_local( Symbol* sym, VMContext* ctx )
{
   return ctx->localVar(sym->m_id);
}

Variable* Symbol::getVariable_closed( Symbol* sym, VMContext* ctx )
{
   return ctx->currentFrame().m_closure->closedData() + sym->m_id;
}

Variable* Symbol::getVariable_extern( Symbol*, VMContext* )
{
   return 0;
}

Variable* Symbol::getVariable_dyns( Symbol* sym, VMContext* ctx )
{
   return ctx->getDynSymbolVariable( sym );
}

Variable* Symbol::setVariable_dyns( Symbol* sym, VMContext* ctx )
{
   return ctx->getLValueDynSymbolVariable( sym );
}

}

/* end of symbol.cpp */
