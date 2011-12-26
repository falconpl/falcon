/*
   FALCON - The Falcon Programming Language.
   FILE: expreeq.cpp

   Syntactic tree item definitions -- Exactly equal (===) expression
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

namespace Falcon {


void ExprEEQ::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprEEQ* self = static_cast<const ExprEEQ*>( ps );
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   // First of all, start executing the start, end and step expressions.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   case 0: 
      // check the start.
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      // fallthrough
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_second, cf ) )
      {
         return;
      }
      // fallthrough
   }
   
   Item *op1, *op2;
   ctx->operands( op1, op2 );

   switch ( op1->type() << 8 | op2->type() )
   {
   case FLC_ITEM_NIL << 8 | FLC_ITEM_NIL:
      op1->setBoolean( true );
      break;

   case FLC_ITEM_BOOL << 8 | FLC_ITEM_BOOL:
      op1->setBoolean( op1->asBoolean() == op2->asBoolean() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      op1->setBoolean( op1->asInteger() == op2->asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      op1->setBoolean( op1->asNumeric() == op2->asNumeric() );
      break;

   case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
      op1->setBoolean( op1->asInst() == op2->asInst() );
      break;

   default:
      op1->setBoolean(false);
   }
   
   ctx->popData();
   ctx->popCode();
}


void ExprEEQ::describeTo( String& ret, int depth ) const
{
   ret = "(" + m_first->describe(depth+1) + " === " + m_second->describe(depth+1) + ")";
}

bool ExprEEQ::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      value.setBoolean( d1.compare(d2) == 0 );
      return true;
   }
   
   return false;
}

}

/* end of expreeq.cpp */