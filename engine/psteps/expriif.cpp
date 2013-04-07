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
#undef SRC
#define SRC "engine/psteps/expriif.cpp"

#include <falcon/psteps/expriif.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

ExprIIF::ExprIIF( int line, int chr ):
   TernaryExpression( line, chr ),
         m_gate( this )
{
   FALCON_DECLARE_SYN_CLASS( expr_iif )
   apply = apply_;
}


ExprIIF::ExprIIF( Expression* op1, Expression* op2, Expression* op3, int line, int chr):
   TernaryExpression( op1, op2, op3 , line, chr),
   m_gate( this )
{
   FALCON_DECLARE_SYN_CLASS( expr_iif )
   apply = apply_;
}
   
ExprIIF::ExprIIF( const ExprIIF& other ): 
   TernaryExpression( other ), 
   m_gate(this)
{
   apply = apply_;
}


ExprIIF::~ExprIIF()
{}

bool ExprIIF::simplify( Item& value ) const
{
   Item temp;
   if( m_first->category() == TreeStep::e_cat_expression
       && m_second->category() == TreeStep::e_cat_expression
       && static_cast<Expression*>(m_first)->simplify( temp ) )
   {
      if (temp.isTrue())
      {
         return static_cast<Expression*>(m_second)->simplify( value );
      }
      else
      {
         return static_cast<Expression*>(m_third)->simplify( value );
      }
   }

   return false;
}


void ExprIIF::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprIIF* self = static_cast<const ExprIIF*>( ps );
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   ctx->resetCode( &self->m_gate );
   // launch the first expression
   if( ctx->stepInYield(self->first()) )
   {
      return;
   }
   
   // we didn't go deep
   ctx->popCode(); // so remove gate
   
   //get the value of the condition and pop it.
   bool cond = ctx->topData().isTrue();
   ctx->popData();
   ctx->stepIn( cond ? self->second() : self->third() );
}

void ExprIIF::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   if( m_first == 0 || m_second == 0 || m_third == 0 )
   {
      tw->write( "/* Blank ExprIIF */" );
   }
   else
   {
      tw->write( "( (" );
      m_first->render( tw, relativeDepth(depth) );
      tw->write( ") ? ");
      m_second->render( tw, relativeDepth(depth) );
      tw->write( " : ");
      m_third->render( tw, relativeDepth(depth) );
      tw->write( " )" );
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

ExprIIF::Gate::Gate( ExprIIF* owner ):
   m_owner(owner) 
{
   apply = apply_;
}

void ExprIIF::Gate::apply_( const PStep* ps,  VMContext* ctx )
{
   const ExprIIF* self = static_cast<const ExprIIF::Gate*>( ps )->m_owner;
   TRACE2( "Apply GATE \"%s\"", ((ExprIIF::Gate*)ps)->describe().c_ize() );

   ctx->popCode();
   
   //get the value of the condition and pop it.
   bool cond = ctx->topData().isTrue();
   ctx->popData();
   ctx->stepIn( cond ? self->second() : self->third() );
}

}

/* end of expriif.cpp */
