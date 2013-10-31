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
#define SRC "engine/psteps/exprin.cpp"

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprin.h>

namespace Falcon {

void ExprIn::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprIn* self = static_cast<const ExprIn*>( ps );
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
      /* no break */
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->second(), cf ) )
      {
         return;
      }
      break;
   }
   
   // we won't be back anymore
   ctx->popCode();

   Class* cls = 0;
   void* data = 0;
   ctx->topData().forceClassInst( cls, data );
   // invert the operands: "x in y" is leaded by y
   ctx->opcodeParam(0).swap(ctx->opcodeParam(1));
   cls->op_in( ctx, data );
}

const String& ExprIn::exprName() const
{
   static String name("in");
   return name;
}


bool ExprIn::simplify( Item& ) const
{
   return false;
}

}

/* end of exprin.cpp */
