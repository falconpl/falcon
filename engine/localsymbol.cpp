/*
   FALCON - The Falcon Programming Language.
   FILE: localsybmol.cpp

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/localsymbol.h>
#include <falcon/stream.h>
#include <falcon/vm.h>
#include <falcon/trace.h>

#include <falcon/psteps/exprsym.h>

namespace Falcon {

LocalSymbol::LocalSymbol( const String& name, int id ):
   Symbol( t_local_symbol, name ),
   m_id( id )
{
}

LocalSymbol::LocalSymbol( const LocalSymbol& other ):
   Symbol( other ),
   m_id( other.m_id )
{
}

LocalSymbol::~LocalSymbol()
{
}


Item* LocalSymbol::value( VMContext* ctx ) const
{
   if( ctx == 0 ) 
   {
      return 0;
   }

   return &ctx->localVar( m_id );
}

void LocalSymbol::apply_( const PStep* s1, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol *>(s1);
   LocalSymbol* sym = static_cast<LocalSymbol*>(self->symbol());
   
   TRACE2( "Apply local '%s'", sym->m_name.c_ize() );
   Item &i = ctx->localVar( sym->m_id );
   ctx->pushData( i );
}


void LocalSymbol::apply_lvalue_( const PStep* s1, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* self = static_cast<const ExprSymbol::PStepLValue *>(s1);
   LocalSymbol* sym = static_cast<LocalSymbol*>(self->m_owner->symbol());
   
   TRACE2( "LValue apply to local '%s'", sym->m_name.c_ize() );
   ctx->localVar( sym->m_id ).assign( ctx->topData() );
   // topData is already the value of the l-value evaluation.
   // so we leave it alone.
}

Expression* LocalSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   sym->setApplyLvalue( apply_lvalue_ );
   return sym;
}

}

/* end of localsymbol.cpp */
