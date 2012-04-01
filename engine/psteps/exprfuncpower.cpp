/*
   FALCON - The Falcon Programming Language.
   FILE: exprfuncpower.cpp

   Syntactic tree item definitions -- Functional power.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 15:49:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprfuncpower.cpp"

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprfuncpower.h>
#include <falcon/classes/classcomposition.h>

namespace Falcon {

void ExprFuncPower::apply_( const PStep* ps, VMContext* ctx )
{
   static class ClassComposition* compo = 
      static_cast<ClassComposition*>(Engine::instance()->getMantra("Composition", Mantra::e_c_class));
   
   const ExprFuncPower* self = static_cast<const ExprFuncPower*>( ps );
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );
   fassert( self->second() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0:
         cf.m_seqId ++;
         if( ctx->stepInYield( self->second() ) )
         {
            return;
         }
         // fallthrough
      case 1:
         cf.m_seqId ++;
         if( ctx->stepInYield( self->first() ) )
         {
            return;
         }
   }
      
   // This evaluate to self.
   void* data = compo->createComposition( ctx->opcodeParam(0), ctx->opcodeParam(1) );
   ctx->popData();
   ctx->topData() = Item( compo, data );
   ctx->popCode();
}


void ExprFuncPower::describeTo( String& ret, int depth ) const
{
   if( m_first == 0 || m_second == 0 )
   {
      ret = "<Blank ExprFuncPower>";
      return;
   }
   
   ret = "(" + m_first->describe(depth+1) + " ^.. " + m_second->describe(depth+1) + ")";
}

bool ExprFuncPower::simplify( Item& ) const
{  
   return false;
}

}

/* end of exprfuncpower.cpp */