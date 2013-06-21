/*
   FALCON - The Falcon Programming Language.
   FILE: treestep.cpp

   PStep that can be inserted in a code tree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 14:38:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/treestep.h>
#include <falcon/class.h>
#include <falcon/item.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/engine.h>
#include <falcon/stdhandlers.h>

#include <falcon/statement.h>
#include <falcon/syntree.h>
#include <falcon/trace.h>


namespace Falcon {

TreeStep::TreeStep( const TreeStep& other ):
   PStep( other ),
   m_handler( other.m_handler ),
   m_parent(0),
   m_pstep_lvalue(0),
   m_cat( other.m_cat ),
   m_bInGC(false)
{
}


TreeStep::~TreeStep()
{
}


void TreeStep::gcMark( uint32 mark )
{
   m_gcMark = mark;
}


void TreeStep::describeTo( String& str ) const
{
   str = m_handler->name();
   str += String().A(" at ").N(sr().line());
}


bool TreeStep::setParent( TreeStep* ts )
{
   TRACE1( "Setting parent of %p to %p (now %p)", this, ts, m_parent );
   
   if( m_parent == 0 || ts == 0 || m_parent == ts )
   {
      m_parent = ts;
      return true;
   }

   return false;
}


int32 TreeStep::arity() const
{
   return 0;
}

TreeStep* TreeStep::nth( int32 ) const
{
   return NULL;
}


bool TreeStep::setNth( int32, TreeStep* )
{
   return false;
}

bool TreeStep::insert( int32, TreeStep* )
{
   return false;
}

bool TreeStep::append( TreeStep* )
{
   return false;
}

bool TreeStep::remove( int32 )
{
   return false;
}

TreeStep* TreeStep::selector() const
{
   return 0;
}

bool TreeStep::selector( TreeStep* )
{
   return false;
}


Expression* TreeStep::checkExpr( const Item& item, bool& bCreate )
{
   Class* cls;
   void* data;
   if( ! item.asClassInst(cls, data) )
   {
      if( bCreate )
      {
         return new ExprValue(item);
      }

      return 0;
   }

   if( item.type() == FLC_CLASS_ID_TREESTEP )
   {
      TreeStep* theStep = static_cast<TreeStep*>( data );
      if( theStep->category() == TreeStep::e_cat_expression )
      {
         bCreate = false;
         return static_cast<Expression*>(theStep);
      }
      return 0;
   }
   else if( item.type() == FLC_CLASS_ID_SYMBOL )
   {
      if( bCreate ) {
         return new ExprSymbol( static_cast<Symbol*>(data) );
      }
      return 0;
   }
   else if( bCreate ) {
      return new ExprValue(item);
   }
   else {
      return 0;
   }
}


Statement* TreeStep::checkStatement( const Item& item )
{
   static Class* clsTreeStep = Engine::handlers()->treeStepClass();

   Class* cls;
   void* data;
   if( ! item.asClassInst(cls, data) )
   {
      return 0;
   }

   //TODO:TreeStepInherit
   if( cls->isDerivedFrom(clsTreeStep) )
   {
      TreeStep* theStep = static_cast<TreeStep*>( data );
      if( theStep->category() == TreeStep::e_cat_statement )
      {
         return static_cast<Statement*>(theStep);
      }
   }
   return 0;
}


SynTree* TreeStep::checkSyntree( const Item& item )
{
   static Class* clsTreeStep = Engine::handlers()->treeStepClass();

   Class* cls;
   void* data;
   if( ! item.asClassInst(cls, data) )
   {
      return 0;
   }

   //TODO:TreeStepInherit
   if( cls->isDerivedFrom(clsTreeStep) )
   {
      TreeStep* theStep = static_cast<TreeStep*>( data );
      if( theStep->category() == TreeStep::e_cat_syntree )
      {
         return static_cast<SynTree*>(theStep);
      }
   }
   return 0;
}

void TreeStep::resolveUnquote( VMContext* ctx, const UnquoteResolver& )
{
   class SelectorUnquoteResolver: public UnquoteResolver
   {
   public:
      SelectorUnquoteResolver( TreeStep* parent ):
         m_parent(parent)
      {}

      virtual ~SelectorUnquoteResolver() {}

      virtual void onUnquoteResolved( TreeStep* newStep ) const {
         m_parent->selector(newStep);
      }

   private:
      TreeStep* m_parent;
   };

   class BranchUnquoteResolver: public UnquoteResolver
   {
   public:
      BranchUnquoteResolver( TreeStep* parent, int32 n ):
         m_parent(parent),
         m_pos(n)
      {}

      virtual ~BranchUnquoteResolver() {}

      virtual void onUnquoteResolved( TreeStep* newStep ) const {
         m_parent->setNth(m_pos,newStep);
      }

   private:
      TreeStep* m_parent;
      int32 m_pos;
   };


   int32 rty = arity();
   TreeStep* sel = selector();
   if( sel != 0 )
   {
      sel->resolveUnquote(ctx, SelectorUnquoteResolver(this) );
   }

   for (int i = rty-1; i >=0; --i ) {
      nth(i)->resolveUnquote(ctx, BranchUnquoteResolver(this, i) );
   }
}


void TreeStep::dispose( TreeStep* child )
{
   if( child != 0 )
   {
      if( child->isInGC() )
      {
         child->setParent(0);
      }
      else
      {
         delete child;
      }
   }
}

}

/* end of treestep.cpp */
