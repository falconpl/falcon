/*
   FALCON - The Falcon Programming Language.
   FILE: exprneg.cpp

   Syntactic tree item definitions -- Numeric unary negator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/psteps/exprneg.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/errors/operanderror.h>

namespace Falcon {


bool ExprNeg::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( -value.asInteger() ); return true;
      case FLC_ITEM_NUM: value.setNumeric( -value.asNumeric() ); return true;
      }
   }

   return false;
}

void ExprNeg::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprNeg* self = static_cast<const ExprNeg*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   fassert( self->first() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      // fallthrough
      case 1:
      {
         cf.m_seqId = 2;
         register Item* item = &ctx->topData();

         switch( item->type() )
         {
            case FLC_ITEM_INT: item->setInteger( -item->asInteger() ); break;
            case FLC_ITEM_NUM: item->setNumeric( -item->asNumeric() ); break;
            default:
            {
               Class* cls;
               void* data;
               if( item->asClassInst( cls, data ) )
               {
                  cls->op_neg( ctx, data );
                  // went deep?
                  if( &cf != &ctx->currentCode() )
                  {
                     // s_nextApply will be called
                     return;
                  }
               }
               else {
                  throw
                     new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("neg") );
               }
            }            
         }
      }
   }
   
   ctx->popCode();     
}


void ExprNeg::describeTo( String& str, int depth ) const
{
   if( m_first == 0 )
   {
      str = "<Blank ExprNeg>";
      return;
   }
   
   str = "(-";
   str += m_first->describe(depth+1)+")";
}

}

/* end of exprneg.cpp */