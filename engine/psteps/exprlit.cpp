/*
   FALCON - The Falcon Programming Language.
   FILE: exprlit.cpp

   Literla expression (*= expr) 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 04 Jan 2012 00:55:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/exprlit.cpp"

#include <falcon/psteps/exprlit.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>

namespace Falcon {

ExprLit::ExprLit():
   UnaryExpression()
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;   
}

ExprLit::ExprLit( Expression* expr ):
   UnaryExpression( expr )
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
}

ExprLit::ExprLit( const ExprLit& other ):
   UnaryExpression( other )
{
   apply = apply_;
}
     
    
void ExprLit::describeTo( String& str, int depth ) const
{
   if( first() == 0 ) {
      str = "<Blank ExprLit>";
      return;
   }
   
   str += "^= " + first()->describe( depth + 1 );
}


void ExprLit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );   
   fassert( self->first() != 0 );
   
   ctx->popCode();
   ctx->pushData( Item( self->first()->cls(), self->first() ) );
}

}

/* end of exprlit.cpp */
