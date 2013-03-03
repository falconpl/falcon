/*
   FALCON - The Falcon Programming Language.
   FILE: exprep.cpp

   Evaluation Parameters expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Jan 2013 13:30:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprep.cpp"

#include <falcon/psteps/exprep.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>

namespace Falcon
{

ExprEP::ExprEP( int line, int chr):
         ExprVector(line,chr)
{
   FALCON_DECLARE_SYN_CLASS( expr_ep );
   apply = apply_;
   m_trait = e_trait_vectorial;
}

ExprEP::ExprEP( const ExprEP& other ):
   ExprVector(other)
{
   FALCON_DECLARE_SYN_CLASS( expr_ep );
   apply = apply_;
   m_trait = e_trait_vectorial;
}


ExprEP::~ExprEP()
{
}

bool ExprEP::simplify( Item& ) const
{
   return false;
}

void ExprEP::describeTo( String& tgt, int depth ) const
{
   //tgt = String(" ").replicate( depthIndent*depth );
   if( arity() == 0 ) {
      tgt = "^()";
      return;
   }

   tgt = "^( ";

   for( int32 count = 0; count < arity(); ++count ) {
      if ( count > 0 ) {
         tgt += ", ";
      }
      tgt+= nth( count )->describe(depth+1);
   }
   tgt += " )";
}

void ExprEP::apply_( const PStep* ps, VMContext* ctx )
{
   ctx->popCode();
   // evaluate to itself
   const ExprEP* ep = static_cast<const ExprEP*>( ps );
   ctx->pushData( Item(ep->handler(), const_cast<ExprEP*>(ep) ) );
}

}

/* end of exprep.cpp */
