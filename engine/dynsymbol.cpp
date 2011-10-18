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

#undef SRC
#define SRC "engine/dynsymbol.cpp"

#include <falcon/trace.h>
#include <falcon/dynsymbol.h>
#include <falcon/vm.h>
#include <falcon/errors/codeerror.h>

#include <falcon/psteps/exprsym.h>

namespace Falcon {

DynSymbol::DynSymbol( const String& name ):
      Symbol( t_dyn_symbol, name )
{
}

DynSymbol::DynSymbol( const DynSymbol& other ):
      Symbol( other )
{
}


DynSymbol::~DynSymbol()
{
}


Item* DynSymbol::value( VMContext* ctx ) const
{
   if( ctx == 0 )
   {
      return 0;
   }

   return ctx->vm()->findLocalItem( name() );
}


void DynSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(ps);
   const DynSymbol* sym = static_cast<const DynSymbol*>(self->symbol());

   Item* fval = ctx->vm()->findLocalItem( sym->name() );
   if ( fval )
   {
      TRACE2( "Apply dynsymbol '%s'", sym->m_name.c_ize() );
      ctx->pushData( *fval );
   }

   //TODO Throw on not found
   //TODO cache if possible.
}


void DynSymbol::apply_lvalue_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* self = static_cast<const ExprSymbol::PStepLValue*>(ps);
   const DynSymbol* sym = static_cast<const DynSymbol*>(self->m_owner->symbol());

   Item* fval = ctx->vm()->findLocalItem( sym->name() );
   if ( fval )
   {
      TRACE2( "LValue apply to dynsymbol '%s'", sym->m_name.c_ize() );
      fval->assign( ctx->topData() );
      // topData is already the value of the l-value evaluation.
      // so we leave it alone.
   }

   //TODO Throw on not found
   //TODO cache if possible.
}


Expression* DynSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   sym->setApplyLvalue( apply_lvalue_ );
   return sym;
}

}

/* end of dynsymbol.h */
