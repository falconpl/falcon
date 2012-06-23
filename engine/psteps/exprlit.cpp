/*
   FALCON - The Falcon Programming Language.
   FILE: exprlit.cpp

   Literal expression (^= expr) 
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

#include <vector>
#include <map>
#include <deque>


namespace Falcon {

class ExprLit::Private
{
public:
   typedef std::vector<Expression*> ExprVector;
   typedef std::map<String,Symbol*> DynSymMap;
   typedef std::deque<GCLock*> LockList;
   
   ExprVector m_exprs;
   DynSymMap m_dynSyms;
   LockList m_locks;
   
   ~Private() {
      LockList::iterator iter = m_locks.begin();
      while( iter != m_locks.end() ) {
         (*iter)->dispose();
         ++iter;
      }
   }
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
   
   str += "{~" + first()->describe( depth ) +"}";
}


void ExprLit::setExpression( Expression* expr )
{
   first(expr);
   _p->m_exprs.clear();
   expr->registerUnquotes(this);
}

void ExprLit::setTree( SynTree*  )
{
   
}

Symbol* ExprLit::makeSymbol( const String& name, int line ) 
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls = Engine::instance()->symbolClass();
   
   Private::DynSymMap::iterator item = _p->m_dynSyms.find(name);
   if( item != _p->m_dynSyms.end() ) {
      return item->second;
   }
   
   Symbol* sym = new Symbol( name, line );
   GCLock* lock = FALCON_GC_STORELOCKED(coll, cls, sym);
   _p->m_locks.push_back( lock );
   _p->m_dynSyms[name] = sym;
   return sym;
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
