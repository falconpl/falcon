/*
   FALCON - The Falcon Programming Language.
   FILE: exproob.cpp

   Syntactic tree item definitions -- Out of band manipulators
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {


bool ExprOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setOob();
      return true;
   }

   return false;
}

void ExprOob::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprOob* self = static_cast<const ExprOob*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
         return;
   }
   
   ctx->topData().setOob();
   ctx->popCode();
}

void ExprOob::describeTo( String& str, int depth ) const
{
   str = "^+";
   str += m_first->describe(depth+1);
}



bool ExprDeoob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.resetOob();
      return true;
   }

   return false;
}

void ExprDeoob::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprDeoob* self = static_cast<const ExprDeoob*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
         return;
   }
   
   ctx->topData().resetOob();
   ctx->popCode();
}

void ExprDeoob::describeTo( String& str, int depth ) const
{
   str = "^-";
   str += m_first->describe(depth+1);
}



bool ExprXorOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.xorOob();
      return true;
   }

   return false;
}

void ExprXorOob::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprXorOob* self = static_cast<const ExprXorOob*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
         return;
   }

   
   ctx->topData().xorOob();
}

void ExprXorOob::describeTo( String& str, int depth ) const
{
   str = "^%";
   str += m_first->describe(depth+1);
}



bool ExprIsOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setBoolean(value.isOob());
      return true;
   }

   return false;
}

void ExprIsOob::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprIsOob* self = static_cast<const ExprIsOob*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
         return;
   }
   
   Item& item = ctx->topData();
   item.setBoolean(item.isOob());
}

void ExprIsOob::describeTo( String& str, int depth ) const
{
   str = "^?";
   str += m_first->describe(depth+1);
}

}

/* end of exproob.cpp */
