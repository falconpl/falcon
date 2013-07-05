/*
   FALCON - The Falcon Programming Language.
   FILE: exprstarindex.cpp

   Syntactic tree item definitions -- Star-index accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprstarindex.cpp"

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/stderrors.h>
#include <falcon/textwriter.h>

#include <falcon/psteps/exprstarindex.h>

namespace Falcon {


void ExprStarIndex::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprStarIndex* self = static_cast<const ExprStarIndex*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );
   fassert( self->second() != 0 );
   
   // manage first and second expression.
   CodeFrame& cs = ctx->currentCode();
   switch( cs.m_seqId )
   {
      case 0:
         cs.m_seqId = 1;
         // first we must resolve the index
         if( ctx->stepInYield( self->second(), cs ) ) {
            return;
         }
         // fallthrough
      case 1:
         cs.m_seqId = 2;
         // then the item on which the index is applied.
         if( ctx->stepInYield( self->first(), cs ) ) {
            return;
         }
         // fallthrough         
   }
   
   // here we're not called anymore.
   ctx->popCode();
   
   Item str = ctx->topData();
   ctx->popData();
   Item& index = ctx->topData();

   if ( str.isString() && index.isOrdinal() )
   {
      index.setInteger( str.asString()->getCharAt((length_t)index.forceInteger()) );
   }
   else
   {
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("[*]") );
   }
  
}


void ExprStarIndex::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_first == 0 || m_second == 0 )
   {
      tw->write("/* Blank ExprStarIndex */" );
   }
   else
   {
      m_first->render(tw, relativeDepth(depth));
      tw->write("[* ");
      m_second->render( tw, relativeDepth(depth) );
      tw->write( "]" );
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


const String& ExprStarIndex::exprName() const
{
   static String name("[*]");
   return name;
}


bool ExprStarIndex::simplify( Item& ) const
{
   return false;
}
 
}

/* end of exprstarindex.cpp */
