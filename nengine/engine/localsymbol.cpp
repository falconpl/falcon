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

#include <falcon/trace.h>

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


void LocalSymbol::assign( VMachine* vm, const Item& value ) const
{
   vm->currentContext()->localVar( m_id ).assign( value );
}


void LocalSymbol::apply_( const PStep* s1, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol *>(s1);
   LocalSymbol* sym = static_cast<LocalSymbol*>(self->symbol());
   
#ifndef NDEBUG
   String name = sym->name();
#endif

   // l-value (assignment)?
   if( self->m_lvalue )
   {
      TRACE2( "LValue apply to local '%s'", name.c_ize() );
      ctx->localVar( sym->m_id ).assign( ctx->topData() );
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }
   else
   {
      // try to load by reference.
      TRACE2( "Apply local '%s'", name.c_ize() );
      Item &i = ctx->localVar( sym->m_id );
      ctx->pushData( i );
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
