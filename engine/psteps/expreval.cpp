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
   
   str += "(^* " + first()->describe( depth + 1 ) + ")";
}


void ExprEval::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* classts = Engine::instance()->treeStepClass();
   
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
   Class* cls = 0;
   void* data = 0;
   // keep the object in the stack.
   ctx->topData().forceClassInst( cls, data );
   
   // should we check for in-context evaluation?
   if( ! ctx->evalOutOfContext() && cls->isDerivedFrom( classts ))
   {
      // no need to check if we're already out of context 
      // -- or if the object is not a class
      TreeStep* ts = static_cast<TreeStep*>(data);
      TreeStep* parent = ts->parent();
      while( parent != 0 )
      {
         if( parent == self )
         {
            // ok, we're for sure in the same context.
            break;
         }
         ts = parent;
         parent = ts->parent();
      }
      
      if( parent == 0 ) 
      {
         // bad sign. let's see if we have a common ancestor.
         parent = self->parent();
         while( parent != ts )
         {
            if( parent == 0 ) {
               // no common ancestor.
               ctx->evalOutOfContext(true);
               ctx->pushCode( &self->m_resetOC );
               // resetOC is a finnaly block.
               ctx->traverseFinally(); 
               break;
            }
            parent = parent->parent();            
         }
      }
   }
   
   // generate evaluation.
   cls->op_eval( ctx, data );
}


void ExprEval::PStepResetOC::apply_(const PStep*, VMContext* ctx)
{
   // for sure, we're done...
   ctx->popCode();
   // and we can reset the OC
   ctx->enterFinally();
   ctx->evalOutOfContext(false);
}

}

/* end of expreval.cpp */
