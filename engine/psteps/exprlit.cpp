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
   Expression( other ),
   m_child(0)
{
   apply = apply_;
   m_trait = e_trait_composite;

   _p = new Private();


   if( other.m_child != 0 ) {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }

   if( other._p->m_exprs.size() ) {
      searchUnquotes( m_child );
   }
}
 
ExprLit::~ExprLit()
{
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

Variable* ExprLit::addParam( const String& name )
{
   if( varmap() == 0 ) {
      setVarMap( new VarMap, true ) ;
   }

   return varmap()->addParam(name);
}

Variable* ExprLit::addLocal( const String& name )
{
   if( varmap() == 0 ) {
      setVarMap( new VarMap, true ) ;
   }

   return varmap()->addLocal(name);
}


int ExprLit::paramCount() const
{
   if( varmap() == 0 ) return 0;
   return varmap()->paramCount();
}


const String& ExprLit::param( int n )
{
   static String none;

   if( varmap() == 0 ){
      return none;
   }

   return varmap()->getParamName(n);
}


void ExprLit::describeTo( String& str, int depth ) const
{
   if( m_child == 0) {
      str = "<Blank ExprLit>";
      return;
   }

   bool isEta = varmap() != 0 && varmap()->isEta();

   const char* etaOpen = isEta ? "[" : "(";
   const char* etaClose = isEta ? "]" : ")";
   str = "{";
   str += etaOpen; 
   
   if( varmap() != 0 && varmap()->paramCount() > 0 )
   {
      for( uint32 i = 0; i < varmap()->paramCount(); ++i ) {
         const String& paramName = varmap()->getParamName(i);
         if( i > 0 ) {
            str += ", ";
         }
         str += paramName;
      }
   }
   str += etaClose;
   
   str += "\n";
   str += m_child->describe(depth+1) + "\n";
   str += String( " " ).replicate( depth * depthIndent ) + "}";
}


void ExprLit::setChild( TreeStep* st )
{
   delete m_child;
   m_child = st;
   st->setParent(this);
}


int32 ExprLit::arity() const {
   return 1;
}
   

TreeStep* ExprLit::nth( int32 n ) const
{
   if( n == 0 ) {
      return m_child;
   }
   return 0;
}
   
bool ExprLit::setNth( int32 n, TreeStep* ts ) 
{
   if( n == 0 ) 
   {
      delete m_child;
      if( ts == 0 ) {
         m_child = 0;
         return true;
      }
      else if( ! ts->parent() ) {
         ts->setParent(this);
         m_child = ts;
         return true;
      }
   }
   
   return false;
}


void ExprLit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );   
   fassert( self->m_child != 0 );
   Private::ExprVector& ev = self->_p->m_exprs;

   CodeFrame& cf = ctx->currentCode();
   // something to be unquoted?
   while (cf.m_seqId < (int) ev.size() )
   {
      if ( ctx->stepInYield( ev[cf.m_seqId ++], cf ) )
      {
         return;
      }
   }

   // ExprLit always evaluate to its child
   ctx->popCode();
   register TreeStep* child = self->child();
   
   if( self->varmap() != 0 ) {
      child->setVarMap(new VarMap(*self->varmap()), true);
   }
   
   TreeStep* nchild = static_cast<TreeStep*>(child->handler()->clone(child));

   if( ev.size() ) {
      nchild->resolveUnquote( ctx );
   }

   ctx->pushData( FALCON_GC_HANDLE( nchild )  );
}

}

/* end of exprlit.cpp */
