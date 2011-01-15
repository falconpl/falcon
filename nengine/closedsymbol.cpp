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

void ClosedSymbol::apply_( const PStep* ps, VMachine* vm )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(ps);
   ClosedSymbol* sym = static_cast<ClosedSymbol*>(self->symbol());

   // l-value (assignment)?
   if( self->m_lvalue )
   {
      sym->m_item = vm->topData();
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      vm->pushData( sym->m_item );
   }
}


Expression* ClosedSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   return sym;
}

}

/* end of closedsymbol.cpp */
