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

#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/stdhandlers.h>


#include <falcon/psteps/exprrange.h>

namespace Falcon
{

ExprRange::ExprRange( int line, int chr ):
   Expression( line, chr ),
   m_estart( 0 ),
   m_eend( 0 ),
   m_estep( 0 )
{
   FALCON_DECLARE_SYN_CLASS( expr_genrange )
   apply = apply_;
}

ExprRange::ExprRange( Expression *estart, Expression* eend, Expression* estep, int line, int chr ):
   Expression( line, chr ),
   m_estart( estart ),
   m_eend( eend ),
   m_estep( estep )
{
   FALCON_DECLARE_SYN_CLASS( expr_genrange )
   apply = apply_;
}


ExprRange::ExprRange( const ExprRange& other ):
   Expression( other ),
   m_estart( 0 ),
   m_eend( 0 ),
   m_estep( 0 )
{
   apply = apply_;
   
   if ( other.m_estart == 0 )
   {
      m_estart = other.m_estart->clone();
      m_estart->setParent(this);
   }

   if ( other.m_eend == 0 )
   {
      m_eend = other.m_eend->clone();
      m_eend->setParent(this);
   }

   if ( other.m_estep == 0 )
   {
      m_estep = other.m_estep->clone();
      m_estep->setParent(this);
   }
}
   

ExprRange::~ExprRange()
{
   dispose( m_estart );
   dispose( m_eend );
   dispose( m_estep );
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
   if( expr != 0 ) {
      if ( ! expr->setParent(this) )
      {
         return;
      }
   }
   dispose( m_estart );
   m_estart = expr;
}


void ExprRange::end( Expression* expr )
{
   if( expr != 0 ) {
      if ( ! expr->setParent(this) )
      {
         return;
      }
   }
   dispose( m_eend );
   m_eend = expr;
}


void ExprRange::step( Expression* expr )
{
   if( expr != 0 ) {
      if ( ! expr->setParent(this) )
      {
         return;
      }
   }
   dispose( m_estep );
   m_estep = expr;
}


bool ExprRange::simplify( Item& ) const
{
   // TODO, create a Proto value?
   return false;
}


int32 ExprRange::arity() const
{
   return 3;
}

TreeStep* ExprRange::nth( int32 n ) const
{
   switch(n)
   {
   case 0: return start();
   case 1: case -2: return end();
   case 2: case -1: return step();
   }

   return 0;
}

bool ExprRange::setNth( int32 n, TreeStep* ts )
{
   if( ts != 0 && (ts->category() != TreeStep::e_cat_expression || ! ts->setParent(this) ) )
   {
      return false;
   }

   Expression* expr = static_cast<Expression*>(ts);

   switch(n)
   {
   case 0: dispose( m_estart ); m_estart = expr; break;
   case 1: case -2: dispose( m_eend ); m_eend = expr; break;
   case 2: case -1: dispose( m_estep ); m_estep = expr; break;
   }

   return true;
}

void ExprRange::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* cls =  Engine::handlers()->rangeClass();   
   
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
   ctx->stackResult( 3, FALCON_GC_STORE( cls, rng ) );
}
 
}

/* end of exprrange.cpp */
