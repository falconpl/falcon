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

#include <vector>
#include <map>
#include <deque>

#include "falcon/syntree.h"
#include "falcon/symboltable.h"


namespace Falcon {

class ExprLit::Private
{
public:
   typedef std::map<String,Symbol*> DynSymMap;
   typedef std::deque<GCLock*> LockList;
   
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
   Expression( line, chr ),
   m_child(0)
{
   _p = new Private;
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;   
}


ExprLit::ExprLit( TreeStep* st, int line, int chr ):
   Expression( line, chr ),
   m_child(st)
{
   _p = new Private;
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   st->setParent(this);
}

ExprLit::ExprLit( const ExprLit& other ):
   Expression( other ),
   m_child(0)
{
   _p = new Private;
   apply = apply_;
   if( other.m_child != 0 ) {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }
}
 
ExprLit::~ExprLit()
{
   delete _p;
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
 

    
void ExprLit::describeTo( String& str, int depth ) const
{
   if( m_child == 0) {
      str = "<Blank ExprLit>";
      return;
   }

   const char* etaOpen = m_paramTable.isEta() ? "[" : "(";
   const char* etaClose = m_paramTable.isEta() ? "]" : ")";
   str = "{";
   str += etaOpen; 
   
   if( m_paramTable.localCount() > 0 ) {
      for( int i = 0; i < m_paramTable.localCount(); ++i ) {
         Symbol* param = m_paramTable.getLocal(i);
         if( i > 0 ) {
            str += ", ";
         }
         str += param->name();
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



void ExprLit::registerUnquote( ExprUnquote* expr )
{
   expr->regID( m_paramTable.localCount() + m_paramTable.closedCount() );
   m_paramTable.addClosed( expr->symbolName() );
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
   static Class* clsClosure = Engine::instance()->closureClass();
   
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );   
   fassert( self->m_child != 0 );
   // ExprLit always evaluate to its child
   ctx->popCode();
   register TreeStep* child = self->child();
   
   if( self->m_paramTable.localCount() + self->m_paramTable.closedCount() ) {
      child->setSymbolTable(const_cast<SymbolTable*>(&self->m_paramTable), false);
   }
   
   if( self->m_paramTable.closedCount() != 0 ) {
      Closure* closure = new Closure( child->handler(), child );
      closure->close(ctx, &self->m_paramTable);
      ctx->pushData(Item(clsClosure, closure));
   }
   else {
      ctx->pushData( Item(child->handler(), child) );
   }
}

}

/* end of exprlit.cpp */
