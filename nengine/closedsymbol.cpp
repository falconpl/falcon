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

namespace Falcon {

ClosedSymbol::ClosedSymbol( const ClosedSymbol& other ):
      Symbol(other),
      m_item( other.m_item )
{}

ClosedSymbol::~ClosedSymbol()
{}

void ClosedSymbol::evaluate( VMachine* vm, Item& value ) const
{
   value = m_item;
}

void ClosedSymbol::leval( VMachine* vm, const Item& assignand, Item& value )
{
   m_item = assignand;
   value = assignand;
}

void ClosedSymbol::serialize( Stream* s ) const
{
   Stream::serialize(s);
   //TODO
}

virtual void ClosedSymbol::deserialize( Stream* s )
{
   Stream::deserialize(s);
   //TODO
}

}

/* end of closedsymbol.cpp */
