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
#include <falcon/textwriter.h>

#include <falcon/psteps/exprdict.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/stdhandlers.h>

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


bool ExprDict::get( size_t n, TreeStep* &first, TreeStep* &second ) const
{
   TreeStepVector_Private::ExprVector& mye = _p->m_exprs;
   if( n < mye.size() )
   {
      first = mye[n/2];
      second = mye[n/2+1];
      return true;
   }

   return false;
}

ExprDict& ExprDict::add( TreeStep* k, TreeStep* v )
{
   _p->m_exprs.push_back( k );
   _p->m_exprs.push_back( v );
   return *this;
}

//=====================================================

void ExprDict::render( TextWriter* tw, int depth ) const
{
   tw->write( renderPrefix(depth) );

   TreeStepVector_Private::ExprVector& mye = _p->m_exprs;

   if( mye.empty() )
   {
     tw->write("[=>]");
   }
   else
   {
      tw->write("[ ");
      TreeStepVector_Private::ExprVector::const_iterator iter = mye.begin();
      while( iter != mye.end() )
      {
        if( iter != mye.begin() )
        {
           tw->write(", ");
        }

        TreeStep* expr = *iter;
        expr->render( tw, relativeDepth(depth) );
        ++iter;
        tw->write( " => " );
        expr = *iter;
        expr->render( tw, relativeDepth(depth) );
        ++iter;
      }
      tw->write( " ]" );
   }

   if( depth < 0 )
   {
      tw->write( "\n" );
   }
}


bool ExprDict::simplify( Item& ) const
{
   return false;
}

//=====================================================

void ExprDict::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* cd_class = Engine::handlers()->dictClass();

   const ExprDict* ea = static_cast<const ExprDict*>(ps);
   
   CodeFrame& cf = ctx->currentCode(); 
   TreeStepVector_Private::ExprVector& mye = ea->_p->m_exprs ;
   TreeStepVector_Private::ExprVector::const_iterator iter = mye.begin() + cf.m_seqId;
   while( iter != mye.end() )
   {
      // generate the expression and eventually yield back.
      cf.m_seqId++;
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

   ctx->stackResult(size, FALCON_GC_STORE(cd_class, nd) );
   
   // we're done.
   ctx->popCode();
}



}

/* end of exprdict.cpp */

