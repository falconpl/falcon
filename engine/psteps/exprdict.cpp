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

#undef SRC
#define SRC "engine/psteps/exprdict.cpp"

#include <falcon/trace.h>
#include <falcon/vm.h>
#include <falcon/engine.h>
#include <falcon/itemdict.h>

#include <falcon/psteps/exprdict.h>

#include <vector>

namespace Falcon
{

class ExprDict::Private {
public:

   typedef std::vector< Expression* > ExprVector;
   ExprVector m_exprs;

   ~Private()
   {
      ExprVector::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete *iter;
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
      Expression* first = *iter;
      ++iter;
      Expression* second = *iter;
      ++iter;
      mye.push_back( std::make_pair(first, second) );
      ++iter;
   }
}

ExprDict::~ExprDict()
{
   delete _p;
}


size_t ExprDict::arity() const
{
   return _p->m_exprs.size()/2;
}


bool ExprDict::get( size_t n, Expression* &first, Expression* &second ) const
{
   Private::ExprVector& mye = _p->m_exprs;
   if( n < mye.size() )
   {
      first = mye[n/2];
      second = mye[n/2+1];
      return true;
   }

   return false;
}

ExprDict& ExprDict::add( Expression* k, Expression* v )
{
   _p->m_exprs.push_back( k );
   _p->m_exprs.push_back( v );
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

      str += (*iter)->describe();
      ++iter;
      str += " => ";
      str += *(iter)->describe();
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

      str += (*iter)->describe();
      ++iter;
      str += " => ";
      str += *(iter)->describe();
      ++iter;
   }

   str += " ]";
}

bool ExprDict::simplify( Item& ) const
{
   return false;
}


//=====================================================

void ExprDict::apply_( const PStep*ps, VMContext* ctx )
{
   static Class* cd_class = Engine::instance()->dictClass();
   static Collector* collector = Engine::instance()->collector();

   const ExprDict* ea = static_cast<const ExprDict*>(ps);
   
   CodeFrame& cf = ctx->currentCode(); 
   Private::ExprVector& mye = ea->_p->m_exprs ;
   Private::ExprVector::const_iterator iter = mye.begin() + cf.m_seqId;
   while( iter != mye.end() )
   {
      // generate the expression and eventually yield back.
      if( ctx->stepInYield( ps, cf ) )
      {
         return;
      }
      ++iter;
   }
   
   size_t size = mye.size();
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
   
   // we're done.
   ctx->popCode();
}



}

/* end of exprdict.cpp */

