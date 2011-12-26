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
#define SRC "engine/psteps/exprindex.cpp"

#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/stdsteps.h>

#include <falcon/psteps/exprindex.h>

namespace Falcon
{

bool ExprIndex::simplify( Item& ) const
{
   //ToDo possibly add simplification for indexing.
   return false;
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


void ExprIndex::describeTo( String& ret, int depth ) const
{
   ret = "(" + m_first->describe(depth+1) + "[" + m_second->describe(depth+1) + "])";
}


bool ExprStarIndex::simplify( Item& ) const
{
   //TODO add simplification for static string star indexing.
   return false;
}

}

/* end of exprindex.cpp */
