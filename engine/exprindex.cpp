/*
   FALCON - The Falcon Programming Language.
   FILE: exprindex.cpp

   Syntactic tree item definitions -- Index accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/exprindex.cpp"

#include <falcon/exprindex.h>
#include <falcon/vmcontext.h>
#include <falcon/pcode.h>
#include <falcon/trace.h>

#include "falcon/stdsteps.h"

namespace Falcon
{

bool ExprIndex::simplify( Item& ) const
{
   //ToDo possibly add simplification for indexing.
   return false;
}

void ExprIndex::precompileLvalue( PCode* pcode ) const
{
   m_first->precompile( pcode );
   m_second->precompile( pcode ); // preparation of the parameters  
   pcode->pushStep( m_pstep_lvalue );  // storage
}


void ExprIndex::precompileAutoLvalue( PCode* pcode, const PStep* activity, bool bIsBinary, bool bSaveOld ) const
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   
   // preparation of the parameters for accessor
   m_first->precompile( pcode );
   m_second->precompile( pcode ); 
   
   // save the accessor parameters and eventually bring in the second parameter.
   if( bIsBinary )
   {
      pcode->pushStep( &steps->m_dupliTop3_ );
   }
   else
   {
      pcode->pushStep( &steps->m_dupliTop2_ );
   }
   
   pcode->pushStep( this ); // get the value at index 
   
   if( bSaveOld )
   {
      // Saves the value.
      pcode->pushStep( &steps->m_copyDown3_ ); 
   }
      
   // Perform the operation
   pcode->pushStep( activity );
   
   // save the value
   pcode->pushStep( &steps->m_swapTopWith2_ ); // restore   
   
   // storage step 
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


void ExprIndex::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprIndex*)ps)->describe().c_ize() );

   Class* cls;
   void* self;
   
   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, self);
   cls->op_getIndex( ctx, self );
}


void ExprIndex::PstepLValue::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   TRACE2( "Apply lvalue \"%s\"", ((ExprIndex::PstepLValue*)ps)->describe().c_ize() );

   Class* cls;
   void* self;
   
   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, self);
   cls->op_setIndex( ctx, self );
}


void ExprIndex::describeTo( String& ret ) const
{
   ret = "(" + m_first->describe() + "[" + m_second->describe() + "])";
}


bool ExprStarIndex::simplify( Item& ) const
{
   //TODO add simplification for static string star indexing.
   return false;
}

}

/* end of exprindex.cpp */
