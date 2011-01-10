/*
   FALCON - The Falcon Programming Language.
   FILE: globalsybmol.cpp

   Syntactic tree item definitions -- expression elements -- symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/globalsymbol.h>
#include <falcon/stream.h>

namespace Falcon {

GlobalSymbol::GlobalSymbol( const String& name, Item* itemPtr ):
      Symbol( t_global_symbol, name ),
      m_itemPtr( itemPtr )
{}

GlobalSymbol::GlobalSymbol( const GlobalSymbol& other ):
      Symbol( other ),
      m_itemPtr( other.m_itemPtr )
{}

GlobalSymbol::~GlobalSymbol()
{}

void GlobalSymbol::perform( VMachine* vm ) const
{
   vm->pushCode( this );
}

void GlobalSymbol::apply( VMachine* vm ) const
{
   // l-value (assignment)?
   if( m_lvalue )
   {
      *m_itemPtr = vm->topData();
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      vm->pushData( *m_itemPtr );
   }
}

void GlobalSymbol::serialize( Stream* s ) const
{
   Symbol::serialize( s );
   //TODO
}

void GlobalSymbol::deserialize( Stream* s )
{
   Symbol::deserialize( s );
   //TODO
}

}

/* end of globalsymbol.cpp */
