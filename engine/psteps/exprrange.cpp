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
#include <falcon/pcode.h>
#include <falcon/stdsteps.h>

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


void ExprRange::serialize( DataWriter* ) const
{
   
}


void ExprRange::deserialize( DataReader* )
{
   
}


   
void ExprRange::precompile( PCode* pcd ) const
{
   static StdSteps* stdsteps = Engine::instance()->stdSteps();
   
   if( m_estart != 0 )
   {
      m_estart->precompile( pcd );
   }
   else
   {
      pcd->pushStep( &stdsteps->m_pushNil_ );
   }

   if( m_eend != 0 )
   {
      m_eend->precompile( pcd );
   }
   else
   {
      pcd->pushStep( &stdsteps->m_pushNil_ );
   }
   
   if( m_estep != 0 )
   {
      m_estep->precompile( pcd );
   }
   else
   {
      pcd->pushStep( &stdsteps->m_pushNil_ );
   }
   
   pcd->pushStep( this );
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


void ExprRange::apply_( const PStep*, VMContext* ctx )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls =  Engine::instance()->rangeClass();   
   
   Range* rng = new Range( 
      ctx->opcodeParam(2).forceInteger(),
      ctx->opcodeParam(1).forceInteger(),
      ctx->opcodeParam(0).forceInteger(),
      ctx->opcodeParam(1).isNil() 
      );
  
   ctx->stackResult( 3, FALCON_GC_STORE( coll, cls, rng ) );
}
 
}

/* end of exprrange.cpp */
