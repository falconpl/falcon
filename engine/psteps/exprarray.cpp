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

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include "exprvector_private.h"

#include <vector>


namespace Falcon
{

ExprArray::ExprArray( int line, int chr ):
   ExprVector( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_genarray )
   apply = apply_;
}


ExprArray::ExprArray( const ExprArray& other ):
   ExprVector(other)
{
   FALCON_DECLARE_SYN_CLASS( expr_genarray )   
   apply = apply_;
}


//=====================================================

void ExprArray::describeTo( String& str, int depth ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin();
   
   if( mye.empty() )
   {
      str = "[]";
      return;
   }
   
   String prefix = String(" ").replicate( (depth+1) * depthIndent );
   str = "[ ";
   while( iter != mye.end() )
   {
      if( str.size() > 2 )
      {
         str += ",\n";
      }

      str += prefix + (*iter)->describe( depth +1 );
      ++iter;
   }

   str += String(" ").replicate( depth * depthIndent ) + "\n] ";
}

void ExprArray::oneLinerTo( String& str ) const
{
   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin();
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
   ExprVector_Private::ExprVector& mye = ea->_p->m_exprs;

   // invoke all the expressions.
   CodeFrame& cf = ctx->currentCode();
   int size = (int) mye.size();
   while( cf.m_seqId < size )
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
