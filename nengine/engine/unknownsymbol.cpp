/*
   FALCON - The Falcon Programming Language.
   FILE: unknownsymbol.cpp

   Syntactic tree item definitions -- expression elements -- symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Apr 2011 21:03:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/unknownsymbol.h>
#include <falcon/exprsym.h>
#include <falcon/trace.h>
#include <falcon/codeerror.h>

#include <vector>

#include <falcon/globalsymbol.h>
#include <falcon/closedsymbol.h>
#include <falcon/dynsymbol.h>
#include <falcon/localsymbol.h>

namespace Falcon {

class UnknownSymbol::Private
{
public:
   typedef std::vector<ExprSymbol*> ExprVector;
   ExprVector m_owners;
};

UnknownSymbol::UnknownSymbol( const String& name ):
   Symbol(t_unknown_symbol, name)
{
   _p = new Private();
}

UnknownSymbol::UnknownSymbol( const UnknownSymbol& other ):
   Symbol(other)
{
   _p = new Private();
}

UnknownSymbol::~UnknownSymbol()
{
   // nothing to do
}

Item* UnknownSymbol::value( VMContext* ) const
{
   throw new CodeError( ErrorParam( e_assign_sym, __LINE__, __FILE__ ).extra(name()) );
}

void UnknownSymbol::apply_( const PStep* s, VMContext* ctx )
{
   const ExprSymbol* self = static_cast<const ExprSymbol*>(s);
   
#ifndef NDEBUG
   Symbol* sym = self->symbol();
   String name = "/* unknown */" + sym->name();
#endif
   TRACE2( "Apply unknown '%s'", name.c_ize() );
   ctx->pushData( Item() ); // a nil
}

#ifndef NDEBUG
void UnknownSymbol::apply_lvalue_( const PStep* s, VMContext* )
{   
   const ExprSymbol::PStepLValue* self = static_cast<const ExprSymbol::PStepLValue*>(s);
   Symbol* sym = self->m_owner->symbol();
   String name = "/* unknown */" + sym->name();
   TRACE2( "LValue apply to unknown symbol '%s'", name.c_ize() );
   // topData is already the value of the l-value evaluation.
   // so we leave it alone.
}
#else
void UnknownSymbol::apply_lvalue_( const PStep*, VMContext* )
{   
}
#endif

Expression* UnknownSymbol::makeExpression()
{
   ExprSymbol* expr = new ExprSymbol(this);
   _p->m_owners.push_back( expr );
   expr->apply = apply_;
   expr->setApplyLvalue( apply_lvalue_ );

   return expr;
}

void UnknownSymbol::define( Symbol* def )
{
   Expression::apply_func af, afl;
   switch(def->type())
   {
      case t_closed_symbol: 
         af = ClosedSymbol::apply_; 
         afl = ClosedSymbol::apply_lvalue_; 
         break;
         
      case t_dyn_symbol: 
         af = DynSymbol::apply_; 
         afl = DynSymbol::apply_lvalue_; 
         break;
         
      case t_global_symbol: 
         af = GlobalSymbol::apply_; 
         afl = GlobalSymbol::apply_lvalue_; 
         break;
         
      case t_local_symbol: 
         af = LocalSymbol::apply_; 
         afl = LocalSymbol::apply_lvalue_; 
         break;
         
      case t_unknown_symbol: 
         af = UnknownSymbol::apply_; 
         afl = UnknownSymbol::apply_lvalue_; 
         break;

      default:
         fassert(0);
         af = 0;
   }

   Private::ExprVector::iterator iter = _p->m_owners.begin();
   while( iter != _p->m_owners.end() )
   {
      ExprSymbol* expr = *iter;
      expr->apply = af;
      expr->setApplyLvalue( afl );
      expr->symbol( def );
      ++iter;
   }
}

}

/* end of unknownsymbol.cpp */
