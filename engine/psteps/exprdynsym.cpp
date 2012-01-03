/*
   FALCON - The Falcon Programming Language.
   FILE: exprdynsym.cpp

   Syntactic tree item definitions -- expression elements -- symbol.

   Pure virtual class base for the various symbol types.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 23:00:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprdynsym.cpp"

#include <falcon/dynsymbol.h>
#include <falcon/vmcontext.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprdynsym.h>

namespace Falcon {

ExprDynSymbol::ExprDynSymbol( int line, int chr ):
   Expression( line, chr ),
   m_symbol( 0 ),
   m_pslv(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_dynsym )
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}


ExprDynSymbol::ExprDynSymbol( DynSymbol* target, int line, int chr ):
   Expression( line, chr ),
   m_symbol( target ),
   m_pslv(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_dynsym )
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}


ExprDynSymbol::ExprDynSymbol( const ExprDynSymbol& other ):
   Expression( other ),
   m_symbol(other.m_symbol),
   m_pslv(this)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_pstep_lvalue->apply = other.m_pstep_lvalue->apply;
}

ExprDynSymbol::~ExprDynSymbol()
{
   // nothig to do
}



void ExprDynSymbol::describeTo( String& val, int ) const
{
   if( m_symbol == 0 )
   {
      val = "<Blank ExprSymbol>";
      return;
   }
   else {   
      val = m_symbol->name();
   }
}



void ExprDynSymbol::PStepLValue::describeTo( String& s, int depth ) const
{
   m_owner->describeTo( s, depth );
}

void ExprDynSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprDynSymbol* es = static_cast<const ExprDynSymbol*>(ps);
   fassert( es->m_symbol != 0 );
   ctx->popCode();
   ctx->pushData( *ctx->getDynSymbolValue(es->m_symbol) );
}


void ExprDynSymbol::PStepLValue::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprDynSymbol::PStepLValue* es = static_cast<const ExprDynSymbol::PStepLValue*>(ps);
   fassert( es->m_owner->m_symbol != 0 );
   ctx->popCode();
   ctx->setDynSymbolValue(es->m_owner->m_symbol, ctx->topData());
}
   
}

/* end of exprdynsym.cpp */
