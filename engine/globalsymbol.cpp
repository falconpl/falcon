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

#undef SRC
#define SRC "engine/globalsymbol.cpp"

#include <falcon/globalsymbol.h>
#include <falcon/vm.h>
#include <falcon/accesserror.h>
#include <falcon/trace.h>

#include <falcon/psteps/exprsym.h>


namespace Falcon {

GlobalSymbol::GlobalSymbol( const String& name, const Item& item ):
      Symbol( t_global_symbol, name ),
      m_item( item )
{
}

GlobalSymbol::GlobalSymbol( const String& name ):
      Symbol( t_global_symbol, name )
{
}



GlobalSymbol::GlobalSymbol( const GlobalSymbol& other ):
      Symbol( other ),
      m_item( other.m_item )
{
}


GlobalSymbol::~GlobalSymbol()
{}

Item* GlobalSymbol::value( VMContext* ) const
{
   return const_cast<Item*>(&m_item);
}


void GlobalSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(ps);
   register GlobalSymbol* sym = static_cast<GlobalSymbol*>(self->symbol());
   TRACE2( "Apply global '%s'", sym->name().c_ize() );
   ctx->pushData( sym->m_item );
}


void GlobalSymbol::apply_lvalue_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* self = static_cast<const ExprSymbol::PStepLValue*>(ps);
   register GlobalSymbol* sym = static_cast<GlobalSymbol*>(self->m_owner->symbol());
   TRACE2( "LValue apply to global '%s'", sym->name().c_ize() );
   sym->m_item.assign( ctx->topData() );
   // topData is already the value of the l-value evaluation.
   // so we leave it alone.
}

Expression* GlobalSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   sym->setApplyLvalue( apply_lvalue_ );
   return sym;
}

}

/* end of globalsymbol.cpp */
