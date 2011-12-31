/*
   FALCON - The Falcon Programming Language.
   FILE: expression.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/trace.h>

namespace Falcon {

Expression::Expression( const Expression &other ):
   TreeStep(other),
   m_pstep_lvalue(0),
   m_trait( other.m_trait )
{}

Expression::~Expression()
{}

//=============================================================

UnaryExpression::UnaryExpression( const UnaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() )
{
}

UnaryExpression::~UnaryExpression()
{
   delete m_first;
}


bool UnaryExpression::isStatic() const
{
   return m_first->isStatic();
}


int32 UnaryExpression::arity() const
{
   return 1;
}


TreeStep* UnaryExpression::nth( int32 n ) const
{
   if( n == 0 || n == -1 )
   {
      return m_first;
   }
   return 0;
}


bool UnaryExpression::nth( int32 n, TreeStep* ts )
{   
   if( n == 0 || n == -1 )
   {
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_first;
      m_first = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}


//=============================================================

BinaryExpression::BinaryExpression( const BinaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() )
{
}

BinaryExpression::~BinaryExpression()
{
   delete m_first;
   delete m_second;
}


bool BinaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic();
}


int32 BinaryExpression::arity() const
{
   return 2;
}


TreeStep* BinaryExpression::nth( int32 n ) const
{
   switch( n )
   {
   case 0: case -2: return m_first;
   case 1: case -1: return m_second;
   }
   return 0;
}


bool BinaryExpression::nth( int32 n, TreeStep* ts )
{
   switch( n )
   {
   case 0: case -2:      
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_first;
      m_first = static_cast<Expression*>(ts);
      return true;
   case 1: case -1:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_second;
      m_second = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}

//=============================================================
TernaryExpression::TernaryExpression( const TernaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() ),
   m_third( other.m_third->clone() )
{
}

TernaryExpression::~TernaryExpression()
{
   delete m_first;
   delete m_second;
   delete m_third;
}

bool TernaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic() && m_third->isStatic();
}


int32 TernaryExpression::arity() const
{
   return 3;
}


TreeStep* TernaryExpression::nth( int32 n ) const
{
   switch( n )
   {
   case 0: case -3: return m_first;
   case 1: case -2: return m_second;
   case 2: case -1: return m_third;
   }
   return 0;
}


bool TernaryExpression::nth( int32 n, TreeStep* ts )
{
   switch( n )
   {
   case 0: case -3:      
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_first;
      m_first = static_cast<Expression*>(ts);
      return true;
   case 1: case -2:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_second;
      m_second = static_cast<Expression*>(ts);
      return true;
   case 2: case -1:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      delete m_third;
      m_third = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}


}

/* end of expression.cpp */
