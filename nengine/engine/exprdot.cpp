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
#define SRC "engine/exprdot.cpp"

#include <falcon/trace.h>
#include <falcon/exprdot.h>
#include <falcon/vmcontext.h>
#include <falcon/pcode.h>
#include <falcon/stdsteps.h>

namespace Falcon
{

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

   Class* cls;
   void* self;
   // get prop name
   const String& prop = dot_lv_expr->m_owner->m_prop;
   //acquire the class
   ctx->topData().forceClassInst(cls, self);
   cls->op_setProperty(ctx, self, prop );
}


void ExprDot::precompileLvalue( PCode* pcode ) const
{
   m_first->precompile( pcode );
   pcode->pushStep( m_pstep_lvalue );
}

void ExprDot::describeTo( String& ret ) const
{
   ret = "(" + m_first->describe() + "." + m_prop + ")";
}


void ExprDot::precompileAutoLvalue( PCode* pcode, const PStep* activity, bool bIsBinary, bool bSaveOld ) const
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   
   // preparation of the parameters for accessor
   m_first->precompile( pcode );
   
   // save the accessor parameters and eventually bring in the second parameter.
   if( bIsBinary )
   {
      pcode->pushStep( &steps->m_dupliTop2_ );
   }
   else
   {
      pcode->pushStep( &steps->m_dupliTop_ );
   }
   
   pcode->pushStep( this ); // get the value at index 
   
   if( bSaveOld )
   {
      // Saves the value.
      pcode->pushStep( &steps->m_copyDown2_ ); 
   }
      
   // Perform the operation
   pcode->pushStep( activity );
   
   // save the value
   pcode->pushStep( &steps->m_swapTop_ ); // restore   
   pcode->pushStep( m_pstep_lvalue );
   
   if( bIsBinary )
   {
      pcode->pushStep( &steps->m_dragDown_ );
   }
   else if( bSaveOld )
   {
      pcode->pushStep( &steps->m_pop_ );
   }
}

}

/* end of exprdot.cpp */
