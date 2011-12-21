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

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

namespace Falcon {


bool ExprIIF::simplify( Item& value ) const
{
   Item temp;
   if( m_first->simplify( temp ) )
   {
      if (temp.isTrue())
      {
         return m_second->simplify( value );
      }
      else
      {
         return m_third->simplify( value );
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

void ExprIIF::describeTo( String& str ) const
{
   str = "( " + m_first->describe() + " ? " + m_second->describe() + " : " + m_third->describe() + " )";
}

ExprIIF::Gate::Gate() {
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
