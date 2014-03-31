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
#include <falcon/textwriter.h>

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
   Expression( line, chr ),
   m_child(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   m_trait = e_trait_composite;
   _p = new Private;
}


ExprLit::ExprLit( TreeStep* st, int line, int chr ):
   Expression( line, chr ),
   m_child(st)
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   st->setParent(this);
   m_trait = e_trait_composite;
   _p = new Private;
}

ExprLit::ExprLit( const ExprLit& other ):
   Expression( other )
{
   apply = apply_;
   m_trait = e_trait_composite;

   _p = new Private();

   if( other.m_child != 0 )
   {
      m_child = other.m_child->clone();
      m_child->setParent(this);
      if( other._p->m_exprs.size() ) {
         searchUnquotes( m_child );
      }
   }
   else {
      m_child = 0;
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
      searchUnquotes(child->nth(i));
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

void ExprLit::setChild( TreeStep* st )
{
   if( st != 0 ) {
      if( ! st->setParent(this) ) {
         return;
      }
   }

   dispose( m_child );
   m_child = st;
}

int32 ExprLit::arity() const
{
   return 1;
}

TreeStep* ExprLit::nth( int32 n ) const
{
   if (n==0 || n == -1) {
      return m_child;
   }
   return 0;
}

bool ExprLit::setNth( int32 n, TreeStep* ts )
{
   if ( (n==0 || n == -1) && ts != 0 && ts->setParent(this) ) {
      dispose( m_child );
      m_child = ts;
      return true;
   }

   return false;
}


void ExprLit::render( TextWriter* tw, int depth ) const
{
   if( m_child == 0 )
   {
      tw->write(renderPrefix(depth));
      tw->write( "/* Blank ExprLit */" );
      if( depth >= 0 )
      {
         tw->write("\n");
      }
   }
   else
   {
      if( m_child->category() != TreeStep::e_cat_expression
               || static_cast<Expression*>(m_child)->trait() != Expression::e_trait_tree )
      {
         // non-tree literals are "real literals"...
         tw->write(renderPrefix(depth));
         tw->write("{[] ");
         m_child->render( tw, depth+1 );
         tw->write("}");
         if( depth >= 0 )
         {
            tw->write("\n");
         }
      }
      else {
         // otherwise, the rendering is fully delegated to the tree child.
         m_child->render( tw, depth );
      }
   }
}


void ExprLit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   fassert( self->child() != 0 );
   Private::ExprVector& ev = self->_p->m_exprs;
   CodeFrame& cf = ctx->currentCode();
   int32& seqId = cf.m_seqId;
   TRACE1( "ExprLit::apply_ %d/%ld \"%s\"", seqId, (long) ev.size(), self->describe().c_ize() );

   // something to be unquoted?
   uint32 evsize = ev.size();
   while (seqId < (int) evsize )
   {
      if ( ctx->stepInYield( ev[seqId++], cf ) )
      {
         return;
      }
   }

   // ExprLit always evaluate to a copy of its child
   ctx->popCode();
   TreeStep* nchild = self->child()->clone();

   class ChildUnquoteResolver: public UnquoteResolver
   {
   public:
      ChildUnquoteResolver( TreeStep*& child ):
         m_child(child)
      {}
      virtual ~ChildUnquoteResolver() {}
      virtual void onUnquoteResolved( TreeStep* newStep ) const
      {
         m_child = newStep;
      }

   private:
      TreeStep*& m_child;
   };

   if( evsize ) {
      nchild->resolveUnquote( ctx, ChildUnquoteResolver(nchild) );
   }

   nchild->setInGC();
   ctx->pushData( FALCON_GC_HANDLE( nchild )  );
}

}

/* end of exprlit.cpp */
