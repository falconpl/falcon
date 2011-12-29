/*
   FALCON - The Falcon Programming Language.
   FILE: exprrange.cpp

   Syntactic tree item definitions -- range generator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 22 Sep 2011 13:26:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprrange.cpp"

#include <falcon/range.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprrange.h>

namespace Falcon
{

ExprRange::ExprRange():
   Expression( t_range ),
   m_estart( 0 ),
   m_eend( 0 ),
   m_estep( 0 )
{
   apply = apply_;
}

ExprRange::ExprRange( Expression *estart, Expression* eend, Expression* estep ):
   Expression( t_range ),
   m_estart( estart ),
   m_eend( eend ),
   m_estep( estep )
{
   apply = apply_;
}


ExprRange::ExprRange( const ExprRange& other ):
   Expression( t_range ),
   m_estart( 0 ),
   m_eend( 0 ),
   m_estep( 0 )
{
   apply = apply_;
   
   if ( other.m_estart == 0 )
   {
      m_estart = other.m_estart->clone();
   }

   if ( other.m_eend == 0 )
   {
      m_eend = other.m_eend->clone();
   }

   if ( other.m_estep == 0 )
   {
      m_estep = other.m_estep->clone();
   }
}
   

ExprRange::~ExprRange()
{
   delete m_estart;
   delete m_eend;
   delete m_estep;
}


void ExprRange::describeTo( String& target, int depth ) const
{
   target = "[";
   if( m_estart != 0 )
   {
      target += m_estart->describe(depth+1);
   }
   target += ":";
   
   if( m_eend != 0 )
   {
      target += m_eend->describe(depth+1);
   }
   
   if( m_estep != 0 )
   {
      target += ":";
      target += m_estep->describe(depth+1);
   }
   
   target += "]";
}


void ExprRange::start( Expression* expr )
{
   delete m_estart;
   m_estart = expr;
}


void ExprRange::end( Expression* expr )
{
   delete m_eend;
   m_eend = expr;
}


void ExprRange::step( Expression* expr )
{
   delete m_estep;
   m_estep = expr;
}


bool ExprRange::simplify( Item& ) const
{
   // TODO, create a Proto value?
   return false;
}


void ExprRange::apply_( const PStep* ps, VMContext* ctx )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls =  Engine::instance()->rangeClass();   
   
   const ExprRange* self = static_cast<const ExprRange*>(ps);
   CodeFrame& cs = ctx->currentCode();
   switch( cs.m_seqId )
   {
   case 0:
      cs.m_seqId = 1;

      if( self->m_estart != 0 ) {
         if( ctx->stepInYield( self->m_estart, cs ) )
         {
            return;
         }
      }
      else {
         ctx->pushData(Item());
      }

      // fallthrough
   case 1:
      cs.m_seqId = 2;

      if( self->m_eend != 0 ) {
         if( ctx->stepInYield( self->m_eend, cs ) )
         {
            return;
         }
      }
      else {
         ctx->pushData(Item());
      }

   case 2:
      cs.m_seqId = 3;

      if( self->m_estep != 0 ) {
         if( ctx->stepInYield( self->m_estep, cs ) )
         {
            return;
         }
      }
      else {
         ctx->pushData(Item());
      }
   }
   
   // TODO: Pool ranges.
   Range* rng = new Range( 
      ctx->opcodeParam(2).forceInteger(),
      ctx->opcodeParam(1).forceInteger(),
      ctx->opcodeParam(0).forceInteger(),
      ctx->opcodeParam(1).isNil() 
      );
  
   // we're done.
   ctx->popCode();
   ctx->stackResult( 3, FALCON_GC_STORE( coll, cls, rng ) );
}
 
}

/* end of exprrange.cpp */
