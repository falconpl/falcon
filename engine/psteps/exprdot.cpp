/*
   FALCON - The Falcon Programming Language.
   FILE: exprdot.cpp

   Syntactic tree item definitions -- Dot accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprdot.cpp"

#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprdot.h>

namespace Falcon
{
ExprDot::ExprDot( const String& prop, Expression* op1, int line, int chr ): 
   UnaryExpression( op1, line, chr ),
   m_pslv(this),
   m_prop(prop)
{
   FALCON_DECLARE_SYN_CLASS( expr_dot )
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}


ExprDot::ExprDot( int line, int chr ): 
   UnaryExpression( line, chr ),
   m_pslv(this)
{ 
   FALCON_DECLARE_SYN_CLASS( expr_dot )
   apply = apply_; 
   m_pstep_lvalue = &m_pslv;
}


ExprDot::ExprDot( const ExprDot& other ):
   UnaryExpression( other ),
   m_pslv(this),
   m_prop(other.m_prop)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}
   
ExprDot::~ExprDot()
{
}

bool ExprDot::simplify( Item& ) const
{
   //ToDo add simplification for known members at compiletime.
   return false;
}


void ExprDot::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprDot*)ps)->describe().c_ize() );
   const ExprDot* dot_expr = static_cast<const ExprDot*>(ps);
   
   fassert( dot_expr->first() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield(dot_expr->m_first, cf ) )
      {
         return;
      }
   }
   
   // anyhow we're done.
   ctx->popCode();
   
   Class* cls;
   void* self;
   // get prop name
   const String& prop = dot_expr->m_prop;
   //acquire the class
   ctx->topData().forceClassInst(cls, self);
   cls->op_getProperty(ctx, self, prop );
}


void ExprDot::PstepLValue::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprDot::PstepLValue* dot_lv_expr = static_cast<const ExprDot::PstepLValue*>(ps);
   TRACE2( "Apply lvalue \"%s\"", dot_lv_expr->m_owner->describe().c_ize() );

   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield(dot_lv_expr->m_owner->m_first, cf ) )
      {
         return;
      }
   }
   
   // anyhow we're done.
   ctx->popCode();
   
   Class* cls;
   void* self;
   // get prop name
   const String& prop = dot_lv_expr->m_owner->m_prop;
   //acquire the class
   ctx->topData().forceClassInst(cls, self);
   cls->op_setProperty(ctx, self, prop );
   // it's not our duty to remove the tompost value from the stack.
}

void ExprDot::describeTo( String& ret, int depth ) const
{
   if( m_first == 0 )
   {
      ret = "<Blank ExprDOT>";
      return;
   }
   
   ret = "(" + m_first->describe(depth+1) + "." + m_prop + ")";
}

}

/* end of exprdot.cpp */
