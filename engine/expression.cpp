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
#include <falcon/textwriter.h>

namespace Falcon {

Expression::Expression( const Expression &other ):
   TreeStep(other),
   m_trait( other.m_trait )
{}

Expression::~Expression()
{}

//=============================================================

UnaryExpression::UnaryExpression( const UnaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() )
{
   m_first->setParent(this);
}

UnaryExpression::~UnaryExpression()
{
   dispose( m_first );
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


bool UnaryExpression::setNth( int32 n, TreeStep* ts )
{   
   if( n == 0 || n == -1 )
   {
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_first );
      m_first = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}


void UnaryExpression::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   if( m_first == 0 )
   {
      tw->write( "/* Blank '" );
      tw->write(exprName());
      tw->write( "' */" );
   }
   else
   {
      tw->write("( ");
      tw->write(exprName());
      tw->write(" ");
      m_first->render( tw, relativeDepth(depth) );
      tw->write(" )");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

//=============================================================

BinaryExpression::BinaryExpression( const BinaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() )
{
   m_first->setParent(this);
   m_second->setParent(this);
}

BinaryExpression::~BinaryExpression()
{
   dispose( m_first );
   dispose( m_second );
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


bool BinaryExpression::setNth( int32 n, TreeStep* ts )
{
   switch( n )
   {
   case 0: case -2:      
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_first );
      m_first = static_cast<Expression*>(ts);
      return true;
   case 1: case -1:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_second );
      m_second = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}

void BinaryExpression::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   if( m_first == 0 || m_second == 0 )
   {
      tw->write( "/* Blank '" );
      tw->write(exprName());
      tw->write( "' */" );
   }
   else
   {
      tw->write("( ");
      m_first->render( tw, relativeDepth(depth) );
      tw->write(" ");
      tw->write(exprName());
      tw->write(" ");
      m_second->render( tw, relativeDepth(depth) );
      tw->write(" )");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}
//=============================================================
TernaryExpression::TernaryExpression( const TernaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() ),
   m_third( other.m_third->clone() )
{
   m_first->setParent(this);
   m_second->setParent(this);
   m_third->setParent(this);
}

TernaryExpression::~TernaryExpression()
{
   dispose( m_first );
   dispose( m_second );
   dispose( m_third );
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


bool TernaryExpression::setNth( int32 n, TreeStep* ts )
{
   switch( n )
   {
   case 0: case -3:      
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_first );
      m_first = static_cast<Expression*>(ts);
      return true;
   case 1: case -2:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_second );
      m_second = static_cast<Expression*>(ts);
      return true;
   case 2: case -1:
      if( ts == 0 || ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) return false;
      dispose( m_third );
      m_third = static_cast<Expression*>(ts);
      return true;
   }
   
   return false;
}

}

/* end of expression.cpp */
