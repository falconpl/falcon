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
#include <falcon/function.h>

namespace Falcon {

Symbol::Symbol( const String& name, type_t t, uint32 id ):
   m_name( name ),
   m_declaredAt(0),
   m_type(t),
   m_external(false),
   m_bConstant(false),
   m_defval(0)
{
   define( t, id );
}      

Symbol::Symbol( const Symbol& other ):
   m_name( other.m_name ),
   m_declaredAt( other.m_declaredAt ),
   m_id( m_id ),
   m_type( other.m_type ),
   m_external(other.m_external),
   m_bConstant(other.m_bConstant),
   m_defval(other.m_defval),

   __value( other.__value )
{}


Symbol::~Symbol()
{
}


void Symbol::define( type_t t, uint32 id )
{
   m_id = id;
   switch(t)
   {
      case e_st_local: __value = value_local; break;
      case e_st_extern: case e_st_global: __value = value_global; break;
      case e_st_closed: __value = value_closed; break;
      case e_st_undefined: __value = value_undef; break;         
   }
}

Item* Symbol::value_global( VMContext*, const Symbol* sym )
{
   return sym->defaultValue();
}

Item* Symbol::value_local( VMContext* ctx, const Symbol* sym )
{
   // the ID of the symbol is relative with respect to stack base, as the parameters.
   return &ctx->localVar( sym->m_id );
}

Item* Symbol::value_closed( VMContext* ctx, const Symbol* sym )
{
   return &ctx->currentFrame().m_function->closedItem( sym->m_id );
}

Item* Symbol::value_undef( VMContext* , const Symbol* )
{
   fassert2( false, "Called value on an undefined symbol");
   return 0;
}

}

/* end of symbol.cpp */
