/*
   FALCON - The Falcon Programming Language.
   FILE: exprbnot.cpp

   Syntactic tree item definitions -- Bitwise not expression
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
#include <falcon/errors/operanderror.h>

namespace Falcon {


bool ExprBNOT::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( ~value.asInteger() ); return true;
      case FLC_ITEM_NUM: value.setInteger( ~value.forceInteger() ); return true;
      }
   }

   return false;
}

void ExprBNOT::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprBNOT* self = static_cast<const ExprBNOT*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
         return;
   }
   
   ctx->popCode();
   Item& item = ctx->topData();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.asInteger() ); break;
      case FLC_ITEM_NUM: item.setInteger( ~item.forceInteger() ); break;
      
      default:
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("^!") );
   }
}

void ExprBNOT::describeTo( String& str, int depth ) const
{
   str = "^! ";
   str += m_first->describe( depth + 1 );
}

}

/* end of exprbnot.cpp */
