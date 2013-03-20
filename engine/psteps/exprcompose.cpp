/*
   FALCON - The Falcon Programming Language.
   FILE: expcompose.cpp

   Syntactic tree item definitions -- Function composition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 15:53:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprcompose.cpp"

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprcompose.h>
#include <falcon/classes/classcomposition.h>

namespace Falcon {

void ExprCompose::apply_( const PStep* ps, VMContext* ctx )
{
   static class ClassComposition* compo = 
      static_cast<ClassComposition*>(Engine::instance()->getMantra("Composition", Mantra::e_c_class));
   
   const ExprCompose* self = static_cast<const ExprCompose*>( ps );
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
         /* no break */
      case 1:
         cf.m_seqId ++;
         if( ctx->stepInYield( self->first() ) )
         {
            return;
         }
         break;
   }
      
   // This evaluate to self.
   void* data = compo->createComposition( ctx->opcodeParam(0), ctx->opcodeParam(1) );
   ctx->popData();
   ctx->topData() = Item( compo, data );
   ctx->popCode();
}


bool ExprCompose::simplify( Item& ) const
{
   return false;
}

const String& ExprCompose::exprName() const
{
   static String name("^.");
   return name;
}

}

/* end of exprcompose.cpp */
