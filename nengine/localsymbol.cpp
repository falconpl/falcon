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

namespace Falcon {

LocalSymbol::LocalSymbol( const LocalSymbol& other ):
   Symbol( other ),
   m_id( other.m_id )
{}

LocalSymbol::~LocalSymbol()
{
}


void LocalSymbol::perform( VMachine* vm ) const
{
   vm->pushCode( this );
}

void LocalSymbol::apply( VMachine* vm ) const
{
   // l-value (assignment)?
   if( m_lvalue )
   {
      vm->localVar( m_id ) = vm->topData();
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      vm->pushData( vm->localVar( m_id ) );
   }
}


void LocalSymbol::serialize( Stream* s ) const
{
   Stream::serialize( s );
   //TODO
}


virtual void LocalSymbol::deserialize( Stream* s )
{
   Stream::deserialize( s );
   //TODO
}

}

/* end of localsymbol.cpp */
