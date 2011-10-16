/*
   FALCON - The Falcon Programming Language.
   FILE: exprlogic.cpp

   Syntactic tree item definitions -- Logic expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 13:39:32 +0200

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

bool ExprNot::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setBoolean( ! value.isTrue() );
      return true;
   }
   return false;
}

void ExprNot::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprNot*)self)->describe().c_ize() );
   ctx->topData().setBoolean( ! ctx->boolTopData() );
   //if the boolTopData goes deep, we'll be called again (and with a bool
   // top data, next time).
}

void ExprNot::describeTo( String& str ) const
{
   str = "not ";
   str += m_first->describe();
}

//=========================================================
//Logic And


bool ExprAnd::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() && si.isTrue() );
      return true;
   }
   return false;

}

void ExprAnd::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   // check the first expression for first...
   m_first->precompile( pcode );
    // add a gate to jump checks on short circuits
   pcode->pushStep( &m_gate );
   // and then the second expr last
   m_second->precompile( pcode );
   pcode->pushStep( this );
   //set shortcircuit jump location to end of expr.
   m_gate.m_shortCircuitSeqId = pcode->size();
  
   
}

void ExprAnd::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprAnd*)self)->describe().c_ize() );

   // use the space left from us by the previous expression
   Item& operand = ctx->topData();

   // Booleanize it
   operand.setBoolean( operand.isTrue() );
}

void ExprAnd::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " and " + m_second->describe() + ")";
}


ExprAnd::Gate::Gate() {
   apply = apply_;
}


void ExprAnd::Gate::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprAnd::Gate*)ps)->describe().c_ize() );

   // read and recycle the topmost data.
   Item& operand = ctx->topData();

   if( ! operand.isTrue() )
   {
      operand.setBoolean( false );
      // pop ourselves and the calling code
      ctx->currentCode().m_seqId = static_cast<const Gate*>(ps)->m_shortCircuitSeqId;
   }
   else {
      // the other expression is bound to push data, remove ours
      ctx->popData();
   }   
}

//=========================================================
//Logic Or

bool ExprOr::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() || si.isTrue() );
      return true;
   }
   return false;
}


void ExprOr::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   // check the first expression for first...
   m_first->precompile( pcode );
   // add a gate to jump checks on short circuits
   pcode->pushStep( &m_gate );
   // and then the second expr last
   m_second->precompile( pcode );
   pcode->pushStep( this );
   //set shortcircuit jump location to end of expr.
   m_gate.m_shortCircuitSeqId = pcode->size();
}


void ExprOr::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprOr*)self)->describe().c_ize() );
   
   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
}


void ExprOr::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " or " + m_second->describe() + ")";
}


ExprOr::Gate::Gate() {
   apply = apply_;
}


void ExprOr::Gate::apply_( const PStep* ps,  VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprOr::Gate*)ps)->describe().c_ize() );

   // read and recycle the topmost data.
   Item& operand = ctx->topData();
   if( operand.isTrue() )
   {
      operand.setBoolean( true );
      // pop ourselves and the calling code
      ctx->currentCode().m_seqId = static_cast<const Gate*>(ps)->m_shortCircuitSeqId;
   }
   else {
      // the other expression is bound to push data, remove ours
      ctx->popData();
   }
}

}

/* end of exprlogic.cpp */
