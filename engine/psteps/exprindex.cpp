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

#include <falcon/synclasses.h>
#include <falcon/engine.h>

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
   
   fassert( ((ExprIndex*)ps)->first() != 0 );
   fassert( ((ExprIndex*)ps)->second() != 0 );
   
   Class* cls;
   void* self;
   
   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, self);
   cls->op_getIndex( ctx, self );
}


void ExprIndex::PstepLValue::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   TRACE2( "Apply lvalue \"%s\"", ((ExprIndex::PstepLValue*)ps)->describe().c_ize() );

   fassert( ((ExprIndex*)ps)->first() != 0 );
   fassert( ((ExprIndex*)ps)->second() != 0 );
   
   Class* cls;
   void* self;
   
   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, self);
   cls->op_setIndex( ctx, self );
}


void ExprIndex::describeTo( String& ret, int depth ) const
{
   if( m_first == 0 || m_second == 0 )
   {
      ret = "<Blank ExprIndex>";
      return;
   }
   
   ret = "(" + m_first->describe(depth+1) + "[" + m_second->describe(depth+1) + "])";
}


bool ExprStarIndex::simplify( Item& ) const
{
   //TODO add simplification for static string star indexing.
   return false;
}

}

/* end of exprindex.cpp */
