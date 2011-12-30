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
#include <falcon/psteps/exprsymbol.h>
#include <falcon/engine.h>

namespace Falcon {


void TreeStep::gcMark( uint32 mark )
{   
   m_gcMark = mark;
}
   
   
bool TreeStep::setParent( TreeStep* ts )
{
   if( m_parent == 0 )
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
   return false;
}
   
   
bool TreeStep::nth( int32, TreeStep* )
{
   return false;
}

bool TreeStep::insert( int32, TreeStep* )
{
   return false;
}

bool TreeStep::remove( int32 )
{
   return false;
}

void TreeStep::selector() const
{
   return 0;
}
   
bool TreeStep::selector( Expression* )
{
   return false;
}


TreeStep* TreeStep::checkExpr( const Item& item, bool& bCreate )
{
   static Class* clsTreeStep = Engine::instance()->treeStepClass();
   static Class* clsSymbol = Engine::instance()->symbolClass();
   
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
   
   //TODO:TreeStepInherit
   if( cls->isDerivedFrom(clsTreeStep) )
   {
      TreeStep* theStep = static_cast<TreeStep*>( data );
      if( theStep->category() == TreeStep::e_cat_expression )
      {
         bCreate = false;
         return theStep;
      }
      return 0;
   }
   else if( cls->isDerivedFrom(clsSymbol) )
   {   
      if( bCreate ) {
         return new ExprSymbol( static_cast<Symbol*>(data) );
      }
      return 0;
   }
   else if( bCreate ) {
      return new ExprSymbol( static_cast<Symbol*>(data) );
   }
   else {
      return 0;
   }
}


TreeStep* TreeStep::checkStatement( const Item& item )
{
   static Class* clsTreeStep = Engine::instance()->treeStepClass();
   
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
         return theStep;
      }
   }
   return 0;
}


TreeStep* TreeStep::checkSyntree( const Item& item )
{
   static Class* clsTreeStep = Engine::instance()->treeStepClass();
   
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
         return theStep;
      }
   }
   return 0;
}


}

/* end of treestep.cpp */
