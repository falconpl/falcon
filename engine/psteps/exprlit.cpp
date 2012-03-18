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

#include <vector>

namespace Falcon {

class ExprLit::Private
{
public:
   typedef std::vector<Expression*> ExprVector;
   ExprVector m_exprs;
};


ExprLit::ExprLit( int line, int chr ):
   UnaryExpression( line, chr ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;   
}


ExprLit::ExprLit( Expression* expr, int line, int chr ):
   UnaryExpression( expr, line, chr ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   if ( expr != 0 ) expr->registerUnquotes( this );
}

ExprLit::ExprLit( const ExprLit& other ):
   UnaryExpression( other ),
   _p( new Private )
{
   apply = apply_;
   if ( first() != 0 ) first()->registerUnquotes( this );
}
     
    
void ExprLit::describeTo( String& str, int depth ) const
{
   if( first() == 0 ) {
      str = "<Blank ExprLit>";
      return;
   }
   
   str += "^= " + first()->describe( depth + 1 );
}


void ExprLit::setExpression( Expression* expr )
{
   first(expr);
   _p->m_exprs.clear();
   expr->registerUnquotes(this);
}


void ExprLit::subscribeUnquote( Expression* expr )
{
   _p->m_exprs.push_back( expr );
}


void ExprLit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );   
   fassert( self->first() != 0 );
   
   TreeStep* child;
   Private::ExprVector& ev = self->_p->m_exprs;
   
   if( ev.empty() ) {
      // Not unquoted expression
      child = self->first();
   }
   else {
      // We have unquoted expressions, so we have to generate them.
      CodeFrame& cf = ctx->currentCode();
      
      while( cf.m_seqId < (int) ev.size() )
      {
         Expression* expr = ev[cf.m_seqId++];
         if( ctx->stepInYield( expr, cf ) )
         {
            return;
         }
      }
      
      // we have the evaluated expressions on the top of the data stack here.
      // the unquoted expressions know that we're trying to clone them...
      child = self->first()->clone();
      ctx->popData( ev.size() );
   }
   
   ctx->popCode();
   ctx->pushData( Item( child->handler(), child ) );   
}

}

/* end of exprlit.cpp */
