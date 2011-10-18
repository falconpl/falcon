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
#include <falcon/pcode.h>
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

void ExprBNOT::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprNeg*)self)->describe().c_ize() );
   
   Item& item = ctx->topData();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.asInteger() ); break;
      case FLC_ITEM_NUM: item.setInteger( ~item.forceInteger() ); break;
      
      default:
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("neg") );
   }
}

void ExprBNOT::describeTo( String& str ) const
{
   str = "^! ";
   str += m_first->describe();
}

}

/* end of exprbnot.cpp */
