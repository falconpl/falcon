/*
   FALCON - The Falcon Programming Language.
   FILE: expriif.cpp

   Syntactic tree item definitions -- Ternary fast-if expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/pcode.h>

namespace Falcon {


bool ExprIIF::simplify( Item& value ) const
{
   Item temp;
   if( m_first->simplify( temp ) )
   {
      if (temp.isTrue())
      {
         return m_second->simplify( value );
      }
      else
      {
         return m_third->simplify( value );
      }
   }

   return false;
}


void ExprIIF::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );
   
   //precompile condition executed first.
   m_first->precompile( pcode );
   //push ourselves to determine where to branch to.
   pcode->pushStep( this );
   //precompile true expr.
   m_second->precompile( pcode );
   //push gate to allow exit after true expr.
   pcode->pushStep( &m_gate );
   //acuire the position of the start of the false expr to jump to.
   m_falseSeqId = pcode->size();
   //precompile false expr.
   m_third->precompile( pcode );
   //acquire the position at the end of the expr to jump over the false expr.
   m_gate.m_endSeqId = pcode->size();
   
}

void ExprIIF::apply_( const PStep* self, VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprIIF*)self)->describe().c_ize() );
   
   //get the value of the condition and pop it.
   Item cond = ctx->topData();
   ctx->popData();

   if ( !cond.isTrue() )
   {
      ctx->currentCode().m_seqId = ((ExprIIF*)self)->m_falseSeqId;
   }

   
}

void ExprIIF::describeTo( String& str ) const
{
   str = "( " + m_first->describe() + " ? " + m_second->describe() + " : " + m_third->describe() + " )";
}

ExprIIF::Gate::Gate() {
   apply = apply_;
}

void ExprIIF::Gate::apply_( const PStep* ps,  VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprIIF::Gate*)ps)->describe().c_ize() );

   ctx->currentCode().m_seqId = ((ExprIIF::Gate*)ps)->m_endSeqId;
}

}

/* end of expriif.cpp */
