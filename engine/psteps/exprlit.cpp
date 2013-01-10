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


namespace Falcon {


ExprLit::ExprLit( int line, int chr ):
   Expression( line, chr ),
   m_child(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   m_trait = e_trait_composite;
}


ExprLit::ExprLit( TreeStep* st, int line, int chr ):
   Expression( line, chr ),
   m_child(st)
{
   FALCON_DECLARE_SYN_CLASS( expr_lit );
   apply = apply_;
   st->setParent(this);
   m_trait = e_trait_composite;
}

ExprLit::ExprLit( const ExprLit& other ):
   Expression( other ),
   m_child(0)
{
   apply = apply_;
   if( other.m_child != 0 ) {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }
   m_trait = e_trait_composite;
}
 
ExprLit::~ExprLit()
{
}


void ExprLit::describeTo( String& str, int depth ) const
{
   if( m_child == 0) {
      str = "<Blank ExprLit>";
      return;
   }

   const char* etaOpen = isEta() ? "[" : "(";
   const char* etaClose = isEta() ? "]" : ")";
   str = "{";
   str += etaOpen; 
   
   if( m_paramTable.paramCount() > 0 )
   {
      for( uint32 i = 0; i < m_paramTable.paramCount(); ++i ) {
         const String& paramName = m_paramTable.getParamName(i);
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



void ExprLit::registerUnquote( ExprUnquote* expr )
{
   expr->regID( m_paramTable.paramCount() + m_paramTable.localCount() + m_paramTable.closedCount() );
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
   const ExprLit* self = static_cast<const ExprLit*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );   
   fassert( self->m_child != 0 );
   // ExprLit always evaluate to its child
   ctx->popCode();
   register TreeStep* child = self->child();
   
   if( self->m_paramTable.paramCount() + self->m_paramTable.localCount() + self->m_paramTable.closedCount() ) {
      child->setVarMap(const_cast<VarMap*>(&self->m_paramTable), false);
   }
   
   ctx->pushData( Item(child->handler(), child) );
}

}

/* end of exprlit.cpp */
