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
#include <falcon/exprsym.h>

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

void LocalSymbol::apply_( const PStep* s1, VMachine* vm )
{
   const ExprSymbol* self = static_cast<const ExprSymbol *>(s1);
   LocalSymbol* sym = static_cast<LocalSymbol*>(self->symbol());

   // l-value (assignment)?
   if( self->m_lvalue )
   {
      vm->localVar( sym->m_id ) = vm->topData();
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      // try to load by reference.
      Item &i = vm->localVar( sym->m_id );
      vm->pushData( i );
   }
}

Expression* LocalSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   return sym;
}

}

/* end of localsymbol.cpp */
