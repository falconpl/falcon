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
#include <falcon/exprsym.h>
#include <falcon/vm.h>

#include <falcon/trace.h>

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


void DynSymbol::assign( VMachine* vm, const Item& value ) const
{
   Item* fval = vm->findLocalItem( name() );
   if( fval == 0 )
   {
      throw new CodeError( ErrorParam( e_undef_sym, __LINE__, __FILE__).extra("Dyn:" + name()));
   }

   fval->assign( value );
}


void DynSymbol::apply_( const PStep* ps, VMachine* vm )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(ps);
   DynSymbol* sym = static_cast<DynSymbol*>(self->symbol());
   register VMContext* ctx = vm->currentContext();

   Item* fval = vm->findLocalItem( sym->name() );
   if ( fval )
   {
      // l-value (assignment)?
      if( self->m_lvalue )
      {
         TRACE2( "LValue apply to dynsymbol '%s'", sym->m_name.c_ize() );
         fval->assign( ctx->topData() );
         // topData is already the value of the l-value evaluation.
         // so we leave it alone.
      }
      else
      {
         TRACE2( "Apply dynsymbol '%s'", sym->m_name.c_ize() );
         ctx->pushData( *fval );
      }
   }

   //TODO Throw on not found
   //TODO cache if possible.
}


Expression* DynSymbol::makeExpression()
{
   ExprSymbol* sym = new ExprSymbol(this);
   sym->setApply( apply_ );
   return sym;
}

}

/* end of dynsymbol.h */
