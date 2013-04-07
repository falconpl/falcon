/*
   FALCON - The Falcon Programming Language.
   FILE: exprparentship.cpp

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 12 Feb 2012 21:56:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprparentship.h>
#include <falcon/synclasses.h>
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

#include "exprvector_private.h"
#include "falcon/psteps/exprinherit.h"

namespace Falcon
{

ExprParentship::ExprParentship( int line, int chr ):
   ExprVector(line, chr)
{
   FALCON_DECLARE_SYN_CLASS( expr_parentship );
   apply = apply_;
}


ExprParentship::ExprParentship( const ExprParentship& other ):
   ExprVector(other)
{
   apply = apply_;
}


ExprParentship::~ExprParentship()
{  
}



void ExprParentship::render( TextWriter* tw, int32 depth ) const
{
   //tw->write( renderPrefix(depth) );

   if( !_p->m_exprs.empty() )
   {
      tw->write( "from \\\n");

      TreeStepVector_Private::ExprVector::const_iterator iter = _p->m_exprs.begin();
      while( _p->m_exprs.end() != iter )
      {
         TreeStep* param = *iter;

         // keep same depth
         tw->write( renderPrefix(depth) );
         param->render( tw, relativeDepth(depth) );
         tw->write( " \\\n" );
         ++iter;
      }
   }
}


void ExprParentship::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprParentship* self = static_cast<const ExprParentship*>(ps);
   
   // we need to "produce" the parameters, were any of it.
   CodeFrame& cf = ctx->currentCode();
   int& seqId = cf.m_seqId;
   const TreeStepVector_Private::ExprVector& exprs = self->_p->m_exprs;
   int size = (int) exprs.size();   
   
   TRACE1("Apply with %d/%d parameters", size-seqId-1, size );
   
   // execute the init of all the stuff ( last to first
   while( seqId < size )
   {
      ++seqId;
      ExprInherit* exp = static_cast<ExprInherit*>(exprs[size - seqId]);
      
      Item& iinst = ctx->self();
      fassert( iinst.isUser() );
      Class* cls = iinst.asClass();
      fassert( cls->isDerivedFrom(exp->base()) );
      void* instance = cls->getParentData( exp->base(), iinst.asInst());
      fassert( instance != 0 );
      
      ctx->pushData( Item( exp->base(), instance) );
   
      if( ctx->stepInYield( exp, cf ) )
      {
         return;
      }
   }
   
   // we're done.
   ctx->popCode();
}


bool ExprParentship::setNth( int32 n, TreeStep* ts )
{
   if( ts == 0 || ts->category() != TreeStep::e_cat_expression ) return false;
   Expression* expr = static_cast<Expression*>(ts);
   if( expr->trait() != Expression::e_trait_inheritance ) return false;   
   return _p->nth(n, expr, this );
}


bool ExprParentship::insert( int32 pos, TreeStep* ts )
{
   if( ts == 0 || ts->category() != TreeStep::e_cat_expression ) return false;
   Expression* expr = static_cast<Expression*>(ts);
   if( expr->trait() != Expression::e_trait_inheritance ) return false;   
   return _p->insert(pos, expr, this );
}

bool ExprParentship::append( TreeStep* ts )
{
   if( ts == 0 || ts->category() != TreeStep::e_cat_expression ) return false;
   Expression* expr = static_cast<Expression*>(ts);
   if( expr->trait() != Expression::e_trait_inheritance ) return false;
   return _p->append( expr, this );
}

}

/* end of exprparentship.cpp */
