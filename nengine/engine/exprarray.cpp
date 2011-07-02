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

#include <falcon/exprarray.h>
#include <falcon/itemarray.h>
#include <falcon/classarray.h>

#include <falcon/pcode.h>
#include <falcon/vm.h>
#include <falcon/engine.h>

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

void ExprArray::describe( String& str ) const
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

void ExprArray::oneLiner( String& str ) const
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

void ExprArray::precompile( PCode* pcd ) const
{
   TRACE3( "Precompiling ExprArray: %p (%s)", pcd, oneLiner().c_ize() );
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();
   while( iter != mye.end() )
   {
      (*iter)->precompile(pcd);
      ++iter;
   }

   pcd->pushStep( this );
}

//=====================================================

void ExprArray::serialize( DataWriter* ) const
{
   // TODO
}

void ExprArray::deserialize( DataReader* )
{
   // TODO
}

bool ExprArray::simplify( Item& ) const
{
   return false;
}

//=====================================================

void ExprArray::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* ca_class = Engine::instance()->arrayClass();
   static Collector* collector = Engine::instance()->collector();

   const ExprArray* ea = static_cast<const ExprArray*>(ps);
   Private::ExprVector& mye = ea->_p->m_exprs;

   ItemArray* array = new ItemArray( mye.size() );
   array->length(mye.size());

   ctx->copyData( array->elements(), mye.size() );
   ctx->popData( mye.size() );
   ctx->pushData( FALCON_GC_STORE( collector, ca_class, array ) );
}



}

/* end of exprarray.cpp */
