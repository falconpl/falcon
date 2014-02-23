/*
   FALCON - The Falcon Programming Language.
   FILE: exprlogic.cpp

   Syntactic tree item definitions -- Logic expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 13:39:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprlogic.cpp"

#include <falcon/psteps/exprlogic.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

bool ExprNot::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setBoolean( ! value.isTrue() );
      return true;
   }
   return false;
}

void ExprNot::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprNot* self = static_cast<const ExprNot*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );

   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
   }

   ctx->topData().setBoolean( ! ctx->boolTopData() );
   ctx->popCode();
}

const String& ExprNot::exprName() const
{
   static String name("not");
   return name;
}

//=========================================================
//Logic And


bool ExprAnd::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() && si.isTrue() );
      return true;
   }
   return false;

}


void ExprAnd::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprAnd* self = static_cast<const ExprAnd*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );
   fassert( self->second() != 0 );

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   // calc first expression
   case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      /* no break */
      // check if the first expression was false and then go away
   case 1:
      if( ! ctx->topData().isTrue() )
      {
         // we're done -- short circuited.
         ctx->topData().setBoolean(false);
         ctx->popCode();
         return;
      }
      ctx->popData();
      // it was true, we need to test the second expression
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_second, cf ) )
      {
         return;
      }
      break;
   }

   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
   ctx->popCode();
}


const String& ExprAnd::exprName() const
{
   static String name("and");
   return name;
}


//=========================================================
//Logic Or

bool ExprOr::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() || si.isTrue() );
      return true;
   }
   return false;
}


void ExprOr::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprOr* self = static_cast<const ExprOr*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   fassert( self->first() != 0 );
   fassert( self->second() != 0 );

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   // calc first expression
   case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      /* no break */

      // check if the first expression was true and then go away
   case 1:
      if( ctx->topData().isTrue() )
      {
         // we're done -- short circuited.
         ctx->topData().setBoolean(true);
         ctx->popCode();
         return;
      }
      // it was false, we need to test the second expression
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_second, cf ) )
      {
         return;
      }
      break;
   }

   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
   ctx->popCode();
}


const String& ExprOr::exprName() const
{
   static String name("or");
   return name;
}

}

/* end of exprlogic.cpp */
