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
   _p = new ExprVector_Private(*other._p, this);
}

ExprVector::~ExprVector()
{
   delete _p;
}

int ExprVector::arity() const
{
   return _p->arity();
}

TreeStep* ExprVector::nth( int32 n ) const
{
   return _p->nth(n);
}

bool ExprVector::setNth( int32 n, TreeStep* ts )
{
   if( ts == 0 || ts->category() != TreeStep::e_cat_expression ) return false;
   return _p->nth(n, static_cast<Expression*>(ts), this );
}

bool ExprVector::insert( int32 n, TreeStep* ts )
{   
   if( ts == 0 || ts->category() != TreeStep::e_cat_expression ) return false;
   return _p->insert(n, static_cast<Expression*>(ts), this );
}

bool ExprVector::append( TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression ) return false;
   return _p->append( static_cast<Expression*>(ts), this );
}

bool ExprVector::remove( int32 n )
{   
   return _p->remove(n);
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


void ExprVector::resolveUnquote( VMContext* ctx )
{
   for ( uint32 i = 0; i < _p->m_exprs.size(); ++i ) {
      _p->m_exprs[i]->resolveUnquote(ctx);
   }
}

}

/* end of exprvector.cpp */
