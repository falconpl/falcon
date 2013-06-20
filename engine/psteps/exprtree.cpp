/*
   FALCON - The Falcon Programming Language.
   FILE: exprtree.h

   Non-syntactic expression holding evaluable code units.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Jan 2013 20:24:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#define SRC "engine/psteps/exprtree.cpp"

#include <falcon/psteps/exprtree.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/symbol.h>
#include <falcon/closure.h>
#include <falcon/symbolmap.h>
#include <falcon/stdsteps.h>
#include <falcon/syntree.h>
#include <falcon/textwriter.h>

#include <set>
#include <vector>


namespace Falcon {

ExprTree::ExprTree( int line, int chr ):
   Expression( line, chr ),
   m_child(0),
   m_symbols(0),
   m_bIsEta(false)
{
   FALCON_DECLARE_SYN_CLASS( expr_tree );
   apply = apply_;
   m_trait = e_trait_tree;
}


ExprTree::ExprTree( TreeStep* st, int line, int chr ):
   Expression( line, chr ),
   m_child(st),
   m_symbols(0),
   m_bIsEta(false)
{
   FALCON_DECLARE_SYN_CLASS( expr_tree );
   apply = apply_;
   m_trait = e_trait_tree;
   st->setParent(this);
}

ExprTree::ExprTree( const ExprTree& other ):
   Expression( other ),
   m_child(0),
   m_symbols(0),
   m_bIsEta(other.m_bIsEta)
{
   apply = apply_;
   m_trait = e_trait_tree;

   if( other.m_child != 0 ) {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }

   if( other.m_symbols != 0 ) {
      m_symbols = new SymbolMap(*other.m_symbols);
   }
}

ExprTree::~ExprTree()
{
   delete m_symbols;
   dispose( m_child );
}


bool ExprTree::addParam( const String& name )
{
   if( m_symbols == 0 ) {
      m_symbols = new SymbolMap;
   }

   return m_symbols->insert(name);
}


int ExprTree::paramCount() const
{
   if( m_symbols == 0 ) return 0;
   return m_symbols->size();
}


const String& ExprTree::param( int n )
{
   static String none;

   if( m_symbols == 0 ){
      return none;
   }

   return m_symbols->getById(n)->name();
}


void ExprTree::render( TextWriter* tw, int32 depth ) const
{
   tw->write(renderPrefix(depth));

   if( m_child == 0) {
      tw->write( "/* Blank ExprLit */");
   }
   else
   {
      bool isEta = m_bIsEta;

      const char* etaOpen = isEta ? "[" : "(";
      const char* etaClose = isEta ? "]" : ")";
      tw->write("{");
      tw->write( etaOpen );

      if( m_symbols != 0 && m_symbols->size() > 0 )
      {
         for( uint32 i = 0; i < m_symbols->size(); ++i ) {
            const String& paramName = m_symbols->getById(i)->name();
            if( i > 0 ) {
               tw->write( ", " );
            }
            tw->write(paramName);
         }
      }
      tw->write( etaClose );

      if ( m_child->category() != TreeStep::e_cat_expression )
      {
         tw->write("\n");
      }
      else {
         tw->write(" ");
      }

      m_child->render( tw, depth+1 );
      if ( m_child->category() != TreeStep::e_cat_expression )
      {
         tw->write("\n");
         tw->write(renderPrefix(depth));
      }

      tw->write("}");
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


void ExprTree::setChild( TreeStep* st )
{
   if( st->setParent(this) )
   {
      dispose( m_child );
      m_child = st;
   }
}


int32 ExprTree::arity() const {
   return 1;
}


TreeStep* ExprTree::nth( int32 n ) const
{
   if( n == 0 || n == -1 ) {
      return m_child;
   }
   return 0;
}

bool ExprTree::setNth( int32 n, TreeStep* ts )
{
   if( n == 0 || n == -1 )
   {
      if( ts == 0 ) {
         dispose( m_child );
         m_child = 0;
         return true;
      }
      else if( ts->setParent(this) ) {
         dispose( m_child );
         m_child = ts;
         return true;
      }
   }

   return false;
}

void ExprTree::setParameters( SymbolMap* vm )
{
   delete m_symbols;
   m_symbols = vm;
}

void ExprTree::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprTree* self = static_cast<const ExprTree*>( ps );
   TRACE1( "ExprTree::apply_ \"%s\"", self->describe().c_ize() );

   // we evaluate to ourselves -- it's the op_call that's different.
   ctx->popCode();
   ctx->pushData( Item(self->handler(), const_cast<ExprTree*>(self)));
}

}

/* end of exprtree.cpp */
