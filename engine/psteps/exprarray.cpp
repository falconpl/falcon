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
#include <falcon/textwriter.h>

#include <falcon/psteps/exprarray.h>
#include <falcon/stdhandlers.h>

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

void ExprArray::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   ExprVector_Private::ExprVector& mye = _p->m_exprs;
   if( mye.empty() )
   {
      tw->write( "[]\n" );
      return;
   }

   tw->write( "[ " );
   bool bFirst = true;
   ExprVector_Private::ExprVector::const_iterator iter = mye.begin();
   while( iter != mye.end() )
   {
      TreeStep* ts = *iter;
      if( bFirst )
      {
         bFirst = false;
      }
      else
      {
         tw->write(", ");
      }
      ts->render( tw, relativeDepth(depth) );

      ++iter;
   }

   tw->write( renderPrefix(depth) );
   tw->write( " ]" );

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


bool ExprArray::simplify( Item& ) const
{
   // TODO: if all expressions are simple, we can create an array.
   return false;
}

//=====================================================

void ExprArray::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* ca_class = Engine::handlers()->arrayClass();
   
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
   ctx->pushData( FALCON_GC_STORE( ca_class, array ) );
   
   ctx->popCode();
}



}

/* end of exprarray.cpp */
