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

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <vector>

#include "exprvector_private.h"

namespace Falcon
{


ExprDict::ExprDict( int line, int chr ):
   ExprVector( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_gendict )
   apply = apply_;
}


ExprDict::ExprDict( const ExprDict& other ):
   ExprVector(other)
{
   apply = apply_;
}

ExprDict::~ExprDict()
{
}


int ExprDict::pairs() const
{
   return (int) _p->m_exprs.size()/2;
}

bool ExprDict::insert( int32, TreeStep* )
{
   return false;
}
   

bool ExprDict::remove( int32 pos )
{
   int size = (int) _p->m_exprs.size()/2;
   if( pos < 0 ) pos = size + pos;
   if( pos < 0 || pos >= size ) return false;
   pos *= 2;
   
   _p->m_exprs.erase(_p->m_exprs.begin()+pos);
   _p->m_exprs.erase(_p->m_exprs.begin()+pos);
   return true;
}


bool ExprDict::get( size_t n, Expression* &first, Expression* &second ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
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

void ExprDict::describeTo( String& str, int depth ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin();

   if( mye.empty() )
   {
      str = "[=>]";
      return;
   }

   String prefix = String(" ").replicate((depth+1) * depthIndent);
   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ",\n";
      }

      str += prefix + (*iter)->describe( depth+1 );
      ++iter;
      str += " => ";
      str += (*iter)->describe( depth+1 );
      ++iter;
   }

   
   str += String(" ").replicate(depth*depthIndent) + "\n]";
}

void ExprDict::oneLinerTo( String& str ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin();

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

      str += (*iter)->oneLiner();
      ++iter;
      str += " => ";
      str += (*iter)->oneLiner();
      ++iter;
   }

   str += " ]";
}

bool ExprDict::simplify( Item& ) const
{
   return false;
}


//=====================================================

void ExprDict::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* cd_class = Engine::instance()->dictClass();
   static Collector* collector = Engine::instance()->collector();

   const ExprDict* ea = static_cast<const ExprDict*>(ps);
   
   CodeFrame& cf = ctx->currentCode(); 
   ExprVector_Private::ExprVector& mye = ea->_p->m_exprs ;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin() + cf.m_seqId;
   while( iter != mye.end() )
   {
      // generate the expression and eventually yield back.
      if( ctx->stepInYield( *iter, cf ) )
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

