/*
   FALCON - The Falcon Programming Language.
   FILE: expreeq.cpp

   Syntactic tree item definitions -- Exactly equal (===) expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprnotin.cpp"

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprnotin.h>

namespace Falcon {

void ExprNotin::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprNotin* self = static_cast<const ExprNotin*>( ps );
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );
   fassert( self->second() != 0 );
   
   // First of all, start executing the start, end and step expressions.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   case 0: 
      // check the start.
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->first(), cf ) )
      {
         return;
      }
      // fallthrough
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->second(), cf ) )
      {
         return;
      }
      // fallthrough
   }
   
   register Item* item = &ctx->topData();
   
   Class* cls;
   void* data;
   item->forceClassInst( cls, data );
   
   cls->op_in( ctx, data );
   // Revert the result of op_in
   Item &res = ctx->topData();
   res.setBoolean( !res.asBoolean() );
   // went deep?
   if( &cf != &ctx->currentCode() )
   {
      // s_nextApply will be called
      return;
   }
   
   ctx->popCode();
}


const String& ExprNotin::exprName() const
{
   static String name("notin");
   return name;
}

bool ExprNotin::simplify( Item& ) const
{
   return false;
}

}

/* end of exprnotin.cpp */
