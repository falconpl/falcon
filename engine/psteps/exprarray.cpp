/*
   FALCON - The Falcon Programming Language.
   FILE: exprarray.cpp

   Syntactic tree item definitions -- array (of) expression(s).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/itemarray.h>
#include <falcon/classes/classarray.h>
#include <falcon/vm.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprarray.h>

#include <vector>

namespace Falcon
{

class ExprArray::Private {
public:

   typedef std::vector< Expression* > ExprVector;
   ExprVector m_exprs;

   ~Private()
   {
      ExprVector::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete (*iter);
         ++iter;
      }
   }
};


ExprArray::ExprArray():
   Expression( t_arrayDecl )
{
   apply = apply_;
   _p = new Private;
}


ExprArray::ExprArray( const ExprArray& other ):
   Expression(other)
{
   _p = new Private;
   Private::ExprVector& oe = other._p->m_exprs;
   Private::ExprVector& mye = _p->m_exprs;

   mye.reserve(oe.size());
   Private::ExprVector::const_iterator iter = oe.begin();
   while( iter != oe.end() )
   {
      mye.push_back( (*iter)->clone() );
      ++iter;
   }
}

ExprArray::~ExprArray()
{
   delete _p;
}


size_t ExprArray::arity() const
{
   return _p->m_exprs.size();
}


Expression* ExprArray::get( size_t n ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   if( n < mye.size() )
   {
      return mye[n];
   }
   
   return 0;
}

ExprArray& ExprArray::add( Expression* e )
{
   _p->m_exprs.push_back(e);
   return *this;
}

//=====================================================

void ExprArray::describeTo( String& str ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();
   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ",\n";
      }

      str += (*iter)->describe();
      ++iter;
   }

   str += "\n]\n";
}

void ExprArray::oneLinerTo( String& str ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();
   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ", ";
      }

      str += (*iter)->oneLiner();
      ++iter;
   }

   str += " ]";
}

//=====================================================

bool ExprArray::simplify( Item& ) const
{
   // TODO: if all expressions are simple, we can create an array.
   return false;
}

//=====================================================

void ExprArray::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* ca_class = Engine::instance()->arrayClass();
   static Collector* collector = Engine::instance()->collector();
   
   const ExprArray* ea = static_cast<const ExprArray*>(ps);
   Private::ExprVector& mye = ea->_p->m_exprs;

   // invoke all the expressions.
   CodeFrame& cf = ctx->currentCode();
   while( cf.m_seqId < mye.size() )
   {
      if( ctx->stepInYield( mye[cf.m_seqId ++], cf ) )
      {
         return;
      }
   }
   
   // we are ready to itemize the expressions.   
   ItemArray* array = new ItemArray( mye.size() );
   array->length(mye.size());

   ctx->copyData( array->elements(), mye.size() );
   ctx->popData( mye.size() );
   ctx->pushData( FALCON_GC_STORE( collector, ca_class, array ) );
   
   ctx->popCode();
}



}

/* end of exprarray.cpp */
