/*
   FALCON - The Falcon Programming Language.
   FILE: exprdict.cpp

   Syntactic tree item definitions -- dictionary def expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/exprdict.h>
#include <falcon/trace.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>
#include <falcon/engine.h>
#include <falcon/itemdict.h>

#include <vector>

namespace Falcon
{

class ExprDict::Private {
public:

   typedef std::vector< std::pair<Expression*, Expression*> > ExprVector;
   ExprVector m_exprs;

   ~Private()
   {
      ExprVector::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete iter->first;
         delete iter->second;
         ++iter;
      }
   }

};


ExprDict::ExprDict():
   Expression( t_dictDecl )
{
   _p = new Private;
   apply = apply_;
}


ExprDict::ExprDict( const ExprDict& other ):
   Expression(other)
{
   _p = new Private;
   apply = apply_;
   Private::ExprVector& oe = other._p->m_exprs;
   Private::ExprVector& mye = _p->m_exprs;

   mye.reserve(oe.size());
   Private::ExprVector::const_iterator iter = oe.begin();
   while( iter != oe.end() )
   {
      mye.push_back( std::make_pair(iter->first->clone(), iter->second->clone()) );
      ++iter;
   }
}

ExprDict::~ExprDict()
{
   delete _p;
}


size_t ExprDict::arity() const
{
   return _p->m_exprs.size();
}


bool ExprDict::get( size_t n, Expression* &first, Expression* &second ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   if( n < mye.size() )
   {
      first = mye[n].first;
      second = mye[n].second;
      return true;
   }

   return false;
}

ExprDict& ExprDict::add( Expression* k, Expression* v )
{
   _p->m_exprs.push_back( std::make_pair(k,v) );
   return *this;
}

//=====================================================

void ExprDict::describeTo( String& str ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();

   if( mye.empty() )
   {
      str = "[=>]";
      return;
   }

   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ",\n";
      }

      str += iter->first->describe();
      str += " => ";
      str += iter->second->describe();
      ++iter;
   }

   str += "\n]\n";
}

void ExprDict::oneLinerTo( String& str ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();

   if( mye.empty() )
   {
      str = "[=>]";
      return;
   }

   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ", ";
      }

      str += iter->first->oneLiner();
      str += " => ";
      str += iter->second->oneLiner();
      ++iter;
   }

   str += " ]";
}

void ExprDict::precompile( PCode* pcd ) const
{
   TRACE3( "Precompiling ExprDict: %p (%s)", pcd, oneLiner().c_ize() );
   Private::ExprVector& mye = _p->m_exprs;
   Private::ExprVector::const_iterator iter = mye.begin();
   while( iter != mye.end() )
   {
      iter->first->precompile(pcd);
      iter->second->precompile(pcd);
      ++iter;
   }

   pcd->pushStep( this );
}


bool ExprDict::simplify( Item& ) const
{
   return false;
}


//=====================================================

void ExprDict::serialize( DataWriter* ) const
{
   // TODO
}

void ExprDict::deserialize( DataReader* )
{
   // TODO
}


//=====================================================

void ExprDict::apply_( const PStep*ps, VMContext* ctx )
{
   static Class* cd_class = Engine::instance()->dictClass();
   static Collector* collector = Engine::instance()->collector();

   const ExprDict* ea = static_cast<const ExprDict*>(ps);
   Private::ExprVector& mye = ea->_p->m_exprs;
   size_t size = mye.size() * 2;
   Item* items = ctx->opcodeParams( size );
   Item* end =  items + size;

   ItemDict* nd = new ItemDict;
   while( items < end )
   {
      Item* key = items++;
      Item* value = items++;
      // items are not marked as copied here.
      nd->insert( *key, *value );
   }
   fassert( items == end );

   ctx->stackResult(size, FALCON_GC_STORE(collector, cd_class, nd) );
}



}

/* end of exprdict.cpp */

