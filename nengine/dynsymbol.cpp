/*
   FALCON - The Falcon Programming Language.
   FILE: dynsymbol.h

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dynsymbol.h>
#include <falcon/vm.h>

namespace Falcon {

DynSymbol::DynSymbol( const DynSymbol& other ):
      Symbol( other )
{
}

DynSymbol::~DynSymbol()
{
}

void DynSymbol::apply( VMachine* vm ) const
{
   Item* fval = vm->findLocalItem( m_name );
   if ( fval )
   {
      // l-value (assignment)?
      if( m_lvalue )
      {
         *fval = vm->topData();
         // topData is already the value of the l-value evaluation.
         // so we leave it alone.
      }
      else
      {
         vm->pushData( *fval );
      }
   }

   //TODO Throw on not found
   //TODO cache if possible.
}


void DynSymbol::serialize( Stream* s ) const
{
   Symbol::serialize( s );
   // TODO
}

void DynSymbol::deserialize( Stream* s )
{
   Symbol::deserialize( s );
   // TODO
}

}

/* end of dynsymbol.h */
