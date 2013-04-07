/*
   FALCON - The Falcon Programming Language.
   FILE: exprmultiunpack.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 14:08:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprmultiunpack.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/itemarray.h>
#include <falcon/symbol.h>
#include <falcon/synclasses_id.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/psteps/exprassign.h>
#include "exprvector_private.h"

namespace Falcon {

//=========================================================
// MultiUnpack
//
ExprMultiUnpack::ExprMultiUnpack( int line, int chr ):
   ExprVector(line, chr)
{
   FALCON_DECLARE_SYN_CLASS( expr_munpack )
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}

ExprMultiUnpack::ExprMultiUnpack( const ExprMultiUnpack& other ):
         ExprVector(other)
{
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}


ExprMultiUnpack::~ExprMultiUnpack()
{
}


bool ExprMultiUnpack::simplify( Item& ) const
{
   return false;
}

bool ExprMultiUnpack::setNth( int32 n, TreeStep* ts )
{
   if ( ts->handler()->userFlags() != FALCON_SYNCLASS_ID_ASSIGN )
   {
      return false;
   }

   return ExprVector::setNth( n, ts );
}


bool ExprMultiUnpack::insert( int32 n, TreeStep* ts )
{
   if ( ts->handler()->userFlags() != FALCON_SYNCLASS_ID_ASSIGN )
   {
      return false;
   }

   return ExprVector::insert( n, ts );
}


bool ExprMultiUnpack::append( TreeStep* ts )
{
   if ( ts->handler()->userFlags() != FALCON_SYNCLASS_ID_ASSIGN )
   {
      return false;
   }

   return ExprVector::append( ts );
}


void ExprMultiUnpack::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( depth < 0 )
   {
      tw->write("(" );
   }

   for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
   {
      ExprAssign* ea = static_cast<ExprAssign*>(_p->m_exprs[i]);
      if ( i >= 1 )
      {
         tw->write(", ");
      }

      ea->first()->render(tw, relativeDepth(depth));
   }

   tw->write( " = " );

   for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
   {
      ExprAssign* ea = static_cast<ExprAssign*>(_p->m_exprs[i]);
      if ( i >= 1 )
      {
         tw->write(", ");
      }

      ea->second()->render(tw, relativeDepth(depth));
   }

   if( depth < 0 )
   {
      tw->write(")" );
   }
   else
   {
      tw->write("\n" );
   }
}



void ExprMultiUnpack::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE3( "Apply multi unpack: %p (%s)", ps, ps->describe().c_ize() );

   ExprMultiUnpack* self = (ExprMultiUnpack*) ps;

   // invoke all the expressions.
   CodeFrame& cf = ctx->currentCode();
   int size = (int) self->_p->m_exprs.size();
   while( cf.m_seqId < size )
   {
      TreeStep* an = self->_p->m_exprs[cf.m_seqId++];
      if( ctx->stepInYield( an, cf ) )
      {
         return;
      }
   }
   
   if( size > 1 ) ctx->popData( size-1 );
   ctx->topData().setNil();
   ctx->popCode();
}



}

/* end of exprmultiunpack.cpp */
