/*
   FALCON - The Falcon Programming Language.
   FILE: expreval.cpp

   Evaluation expression (^* expr) 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 20:02:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/expreval.cpp"

#include <falcon/psteps/expreval.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>

namespace Falcon {

ExprEval::ExprEval():
   UnaryExpression()
{
   FALCON_DECLARE_SYN_CLASS( expr_eval );
   apply = apply_;   
}

ExprEval::ExprEval( Expression* expr ):
   UnaryExpression( expr )
{
   FALCON_DECLARE_SYN_CLASS( expr_eval );
   apply = apply_;
}

ExprEval::ExprEval( const ExprEval& other ):
   UnaryExpression( other )
{
   apply = apply_;
}
     
    
void ExprEval::describeTo( String& str, int depth ) const
{
   if( first() == 0 ) {
      str = "<Blank ExprEval>";
      return;
   }
   
   str += "^* " + first()->describe( depth + 1 );
}


void ExprEval::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprEval* self = static_cast<const ExprEval*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );
   
   fassert( self->first() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   // generate the first?
   if( cf.m_seqId == 0 )
   {
      // yep, generate it
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->first(), cf ) )
      {
         return;
      }
   }
   
   // we're done...
   ctx->popCode();
   // ... get the generated item and call evaluate on it.
   Class* cls;
   void* data;
   // keep the object in the stack.
   ctx->topData().forceClassInst( cls, data );
   // generate evaluation.
   cls->op_eval( ctx, data );
}

}

/* end of expreval.cpp */
