/*
   FALCON - The Falcon Programming Language.
   FILE: exprvector.h

   Common interface for expressions holding vector of expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 12:51:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/psteps/exprvector.h>
#include "exprvector_private.h"
#include "falcon/psteps/exprvalue.h"

namespace Falcon
{  

ExprVector::ExprVector():
   Expression()
{
   _p = new ExprVector_Private;
}


ExprVector::ExprVector( int line, int chr):
   Expression(line, chr)
{
   _p = new ExprVector_Private;
}


ExprVector::ExprVector( const ExprVector& other ):
   Expression(other)
{
   _p = new ExprVector_Private;
   ExprVector_Private::ExprVector& oe = other._p->m_exprs;
   ExprVector_Private::ExprVector& mye = _p->m_exprs;

   mye.reserve(oe.size());
   ExprVector_Private::ExprVector::const_iterator iter = oe.begin();
   while( iter != oe.end() )
   {
      mye.push_back( (*iter)->clone() );
      ++iter;
   }
}

ExprVector::~ExprVector()
{
   delete _p;
}

int ExprVector::arity() const
{
   return (int) _p->m_exprs.size();
}

TreeStep* ExprVector::nth( int32 n ) const
{
   if( n < 0 ) n = (int) _p->m_exprs.size() + n;
   if( n < 0 || n >= _p->m_exprs.size() ) return 0;
   
   return (int) _p->m_exprs[n];
}

bool ExprVector::nth( int32 n, TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression ) return false;
   if( n < 0 ) n = (int) _p->m_exprs.size() + n;
   if( n < 0 || n >= _p->m_exprs.size() ) return false;
   if( ! ts->setParent(this) ) return false;
   
   delete _p->m_exprs[n];
   _p->m_exprs[n] = static_cast<Expression*>(ts);
   return true;
}

bool ExprVector::insert( int32 n, TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression ) return false;
   if( ! ts->setParent(this) ) return false;
   
   if( n < 0 ) n = (int) _p->m_exprs.size() + n;
   
   if( n < 0 || n >= _p->m_exprs.size() ) {      
      _p->m_exprs.push_back(static_cast<Expression*>(ts));
   }
   else {
      delete _p->m_exprs[n];
      _p->m_exprs[n] = static_cast<Expression*>(ts);
   }
   
   return true;
}

bool ExprVector::remove( int32 n )
{   
   if( n < 0 ) n = (int) _p->m_exprs.size() + n;
   
   if( n < 0 || n >= _p->m_exprs.size() ) {      
      return false;
   }
   
   delete _p->m_exprs[n];
   _p->m_exprs.erase( _p->m_exprs.begin() + n );
}


Expression* ExprVector::get( size_t n ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   if( n < mye.size() )
   {
      return mye[n];
   }
   
   return 0;
}

ExprVector& ExprVector::add( Expression* e )
{
   if( e->setParent(this) )
   {
      _p->m_exprs.push_back(e);
   }
   return *this;
}

}

/* end of exprvector.cpp */
