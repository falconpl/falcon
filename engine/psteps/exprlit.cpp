/*
   FALCON - The Falcon Programming Language.
   FILE: exprlit.cpp

   Literal expression {(..) expr } 
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
#include <falcon/symbol.h>
#include <falcon/gclock.h>
#include <falcon/closure.h>

#include <set>
#include <falcon/syntree.h>

#include <vector>

namespace Falcon {

class ExprLit::Private
{
public:
   typedef std::vector<Expression*> ExprVector;

   ExprVector m_exprs;

   Private() {}
   Private( const Private& other ):
      m_exprs(other.m_exprs)
   {}

   ~Private() {}
};

ExprLit::ExprLit( int line, int chr ):
   UnaryExpression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   m_trait = e_trait_composite;
   _p = new Private;
}


ExprLit::ExprLit( Expression* st, int line, int chr ):
   UnaryExpression( st, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   st->setParent(this);
   m_trait = e_trait_composite;
   _p = new Private;
}

ExprLit::ExprLit( const ExprLit& other ):
   UnaryExpression( other )
{
   apply = apply_;
   m_trait = e_trait_composite;

   _p = new Private();

   if( other._p->m_exprs.size() ) {
      searchUnquotes( m_first );
   }
}
 
ExprLit::~ExprLit()
{
   delete _p;
}


void ExprLit::searchUnquotes( TreeStep* child )
{
   if( child->category() == TreeStep::e_cat_expression ) {
      Expression* expr = static_cast<Expression*>(child);

      if( expr->trait() == Expression::e_trait_unquote ) {
         _p->m_exprs.push_back(expr);
         return;
      }
      // don't saearch for unquotes in other sub-lits
      else if( expr->handler() == handler() ) {
         return;
      }
   }

   for( int i = 0; i < child->arity(); ++i ) {
      searchUnquotes(child);
   }
}

void ExprLit::registerUnquote( Expression* unquoted )
{
   _p->m_exprs.push_back(unquoted);
}

uint32 ExprLit::unquotedCount()
{
   return _p->m_exprs.size();
}


Expression* ExprLit::unquoted( uint32 i )
{
   return _p->m_exprs[i];
}

void ExprLit::describeTo( String& str, int depth ) const
{
   if( m_first == 0) {
      str = "<Blank ExprLit>";
      return;
   }
   
   // we're transparent
   m_first->describeTo(str, depth);
}

void ExprLit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   fassert( self->first() != 0 );
   Private::ExprVector& ev = self->_p->m_exprs;
   CodeFrame& cf = ctx->currentCode();
   int32& seqId = cf.m_seqId;
   TRACE1( "ExprLit::apply_ %d/%d \"%s\"", seqId, ev.size(), self->describe().c_ize() );

   // something to be unquoted?
   while (seqId < (int) ev.size() )
   {
      if ( ctx->stepInYield( ev[seqId++], cf ) )
      {
         return;
      }
   }

   // ExprLit always evaluate to a copy of its child
   ctx->popCode();
   Expression* nchild = self->first()->clone();

   if( ev.size() ) {
      nchild->resolveUnquote( ctx );
   }

   ctx->pushData( FALCON_GC_HANDLE( nchild )  );
}

}

/* end of exprlit.cpp */
