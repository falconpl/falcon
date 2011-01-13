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

namespace Falcon {

ClosedSymbol::ClosedSymbol( const ClosedSymbol& other ):
      Symbol(other),
      m_item( other.m_item )
{}

ClosedSymbol::~ClosedSymbol()
{}

void ClosedSymbol::apply( VMachine* vm ) const
{
   // l-value (assignment)?
   if( m_lvalue )
   {
      m_item = vm->topData();
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      vm->pushData( m_item );
   }
}


void ClosedSymbol::serialize( Stream* s ) const
{
   Symbol::serialize(s);
   //TODO
}

void ClosedSymbol::deserialize( Stream* s )
{
   Symbol::deserialize(s);
   //TODO
}

}

/* end of closedsymbol.cpp */
