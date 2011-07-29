/*
   FALCON - The Falcon Programming Language.
   FILE: closedsymbol.cpp

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/closedsymbol.h>
#include <falcon/stream.h>
#include <falcon/vm.h>
#include <falcon/exprsym.h>

#include <falcon/trace.h>

namespace Falcon {

ClosedSymbol::ClosedSymbol( const String& name, const Item& closed ):
   Symbol( t_closed_symbol, name ),
   m_item( closed )
{
}

ClosedSymbol::ClosedSymbol( const ClosedSymbol& other ):
      Symbol(other),
      m_item( other.m_item )
{
}

ClosedSymbol::~ClosedSymbol()
{}

void ClosedSymbol::assign( VMContext*, const Item& value ) const
{
   const_cast<ClosedSymbol*>(this)->m_item.assign( value );
}

 bool ClosedSymbol::retrieve( Item& value, VMContext* ) const
 {
    value = const_cast<ClosedSymbol*>(this)->m_item;
    return true;
 }

void ClosedSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(ps);
   ClosedSymbol* sym = static_cast<ClosedSymbol*>(self->symbol());

   TRACE2( "Apply closed '%s'", sym->m_name.c_ize() );
   ctx->pushData( sym->m_item );
}



void ClosedSymbol::apply_lvalue_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* self = static_cast<const ExprSymbol::PStepLValue*>(ps);
   ClosedSymbol* sym = static_cast<ClosedSymbol*>(self->m_owner->symbol());

   TRACE2( "LValue apply to closed '%s'", sym->m_name .c_ize() );
   sym->m_item.assign( ctx->topData() );
   // topData is already the value of the l-value evaluation.
   // so we leave it alone.
}


Expression* ClosedSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   sym->setApplyLvalue( apply_lvalue_ );
   return sym;
}

}

/* end of closedsymbol.cpp */
