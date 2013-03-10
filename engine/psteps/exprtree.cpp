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
#include <falcon/varmap.h>
#include <falcon/stdsteps.h>
#include <falcon/syntree.h>

#include <set>
#include <vector>


namespace Falcon {

ExprTree::ExprTree( int line, int chr ):
   Expression( line, chr ),
   m_child(0),
   m_varmap(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_tree );
   apply = apply_;
}


ExprTree::ExprTree( TreeStep* st, int line, int chr ):
   Expression( line, chr ),
   m_child(st),
   m_varmap(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_tree );
   apply = apply_;
   st->setParent(this);
}

ExprTree::ExprTree( const ExprTree& other ):
   Expression( other ),
   m_child(0),
   m_varmap(0)
{
   apply = apply_;
   m_trait = e_trait_composite;

   if( other.m_child != 0 ) {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }

   if( other.m_varmap != 0 ) {
      m_varmap = new VarMap(*other.m_varmap);
   }
}

ExprTree::~ExprTree()
{
   delete m_varmap;
   dispose( m_child );
}


Variable* ExprTree::addParam( const String& name )
{
   if( varmap() == 0 ) {
      m_varmap = new VarMap;
   }

   return varmap()->addParam(name);
}

Variable* ExprTree::addLocal( const String& name )
{
   if( varmap() == 0 ) {
      m_varmap = new VarMap;
   }

   return varmap()->addLocal(name);
}


int ExprTree::paramCount() const
{
   if( varmap() == 0 ) return 0;
   return varmap()->paramCount();
}


const String& ExprTree::param( int n )
{
   static String none;

   if( varmap() == 0 ){
      return none;
   }

   return varmap()->getParamName(n);
}


void ExprTree::describeTo( String& str, int depth ) const
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

   if ( m_child->category() != TreeStep::e_cat_expression )
   {
      str += "\n";
   }
   else {
      str += " ";
   }

   str += m_child->describe(depth+1);
   if ( m_child->category() != TreeStep::e_cat_expression )
   {
      str += "\n";
      str += String( " " ).replicate( depth * depthIndent ) + "}";
   }
   else {
      str += "}";
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

void ExprTree::setVarMap( VarMap* vm )
{
   delete m_varmap;
   m_varmap = vm;
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
