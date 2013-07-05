/*
   FALCON - The Falcon Programming Language.
   FILE: exprunpack.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 13:39:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprunpack.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/stderrors.h>
#include <falcon/itemarray.h>
#include <falcon/symbol.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include "exprvector_private.h"

namespace Falcon {



//=========================================================
// Unpack

ExprUnpack::ExprUnpack( Expression* op1, int line, int chr ):
   ExprVector(line, chr),
   m_expander(op1)
{
   FALCON_DECLARE_SYN_CLASS( expr_unpack )
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}

ExprUnpack::ExprUnpack( int line, int chr ):
   ExprVector(line, chr),
   m_expander(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_unpack )
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}

ExprUnpack::ExprUnpack( const ExprUnpack& other ):
   ExprVector(other)
{
   m_trait = Expression::e_trait_composite;
   m_expander = other.m_expander->clone();
   m_expander->setParent(this);
}

ExprUnpack::~ExprUnpack()
{
   dispose( m_expander );
}


bool ExprUnpack::selector( TreeStep* sel )
{
   if( sel->setParent(this) ) {
      dispose(m_expander);
      m_expander = sel;
      return true;
   }

   return false;
}


bool ExprUnpack::simplify( Item& ) const
{
   return false;
}


bool ExprUnpack::setNth( int32 n, TreeStep* ts )
{
   // we accept assignable expressions only
   Expression* temp = static_cast<Expression*>(ts);
   if( ts->category() != TreeStep::e_cat_expression || temp->lvalueStep() == 0 )
   {
      return false;
   }

   return ExprVector::setNth(n,  ts);
}


bool ExprUnpack::insert( int32 n, TreeStep* ts )
{
   Expression* temp = static_cast<Expression*>(ts);
   if( ts->category() != TreeStep::e_cat_expression || temp->lvalueStep() == 0 )
   {
      return false;
   }

   return ExprVector::insert(n, ts);
}


bool ExprUnpack::append( TreeStep* ts )
{
   Expression* temp = static_cast<Expression*>(ts);
   if( ts->category() != TreeStep::e_cat_expression || temp->lvalueStep() == 0 )
   {
      return false;
   }

   return ExprVector::append(ts);
}


void ExprUnpack::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_expander == 0 )
   {
      tw->write("/* Blank ExprUnpack */" );
   }
   else
   {

      // and generate all the expressions, in inverse order.
      for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
      {
         if ( i > 0 )
         {
            tw->write(", ");
         }
         _p->m_exprs[i]->render(tw, relativeDepth(depth));
      }

      tw->write( " = " );
      m_expander->render( tw, relativeDepth(depth) );
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


void ExprUnpack::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE3( "Apply unpack: %p (%s)", ps, ps->describe().c_ize() );

   const ExprUnpack* self = static_cast<const ExprUnpack*>(ps);
   int pcount = (int) self->_p->m_exprs.size();
   
   // eventually generate the expander.
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_expander, cf ) )
      {
         return;
      }
   }
   
   // we won't be called anymore
   register Item& expander = ctx->topData();
   if ( ! expander.isArray() )
   {
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_unpack_size, __LINE__ ).extra("Not an array") );
   }
   ItemArray& array = *(ItemArray*) expander.asInst();

   if( pcount != (int) array.length() )
   {
      throw
         new OperandError( ErrorParam(e_unpack_size, __LINE__ ).extra("Different size") );
   }

   while( cf.m_seqId <= pcount )
   {
      // we start from 1...
      register int pos = cf.m_seqId-1;
      TreeStep* expr = self->_p->m_exprs[pos];
      ctx->pushData( array[pos] );
      fassert( expr->lvalueStep() != 0 );
      cf.m_seqId++;

      if( ctx->stepInYield( expr->lvalueStep(), cf ) )
      {
         return;
      }
   }

   if( pcount > 1) ctx->popData(pcount-1);
   ctx->topData().setNil();

   ctx->popCode();
   // leave the expander in the stack.
}

}

/* end of exprunpack.cpp */
