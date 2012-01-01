/*
   FALCON - The Falcon Programming Language.
   FILE: exprclosure.h

   Genearte a closure out of a function value.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 21:15:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/exprclosure.cpp"

#include <falcon/psteps/exprclosure.h>
#include <falcon/closure.h>
#include <falcon/function.h>
#include <falcon/synclasses.h>

#include "falcon/vmcontext.h"

namespace Falcon {

ExprClosure::ExprClosure():
   m_function(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_closure );
   apply = apply_;   
}


ExprClosure::ExprClosure( Function* closed ):
   m_function(closed)
{
   FALCON_DECLARE_SYN_CLASS( expr_closure );
   apply = apply_;   
   
   // TODO: GarbageLock the function
}

ExprClosure::ExprClosure( const ExprClosure& other ):
   Expression(other),
   m_function( other.m_function )
{
   apply = apply_;
   // TODO: GarbageLock the function
}

ExprClosure::~ExprClosure() {}
   

void ExprClosure::describeTo( String& tgt, int ) const
{
   if( m_function == 0 ) {
      tgt = "<Blank ExprClosure>";
      return;
   }
   tgt = "/* close */ " + m_function->name();
}
   
   
void ExprClosure::apply_( const PStep* ps, VMContext* ctx )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* closureClass = Engine::instance()->closureClass();
   
   const ExprClosure* self = static_cast<const ExprClosure*>(ps);
   fassert( self->m_function != 0 );
   
   // Around just once
   ctx->popCode();
   
   if( self->m_function->symbols().closedCount() > 0 ) {
      // Create the closure and close it.
      Closure* cls = new Closure( self->m_function );
      cls->close( ctx );

      // and return it.
      ctx->pushData( FALCON_GC_STORE( coll , closureClass, cls ) );         
   }
   else {
      ctx->pushData( self->m_function );
   }
}

}

/* end of exprclosure.cpp */
