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
#include <falcon/pcode.h>

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

void ExprOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprOob*)self)->describe().c_ize() );
   
   ctx->topData().setOob();
}

void ExprOob::describeTo( String& str ) const
{
   str = "^+";
   str += m_first->describe();
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

void ExprDeoob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprDeoob*)self)->describe().c_ize() );
   
   ctx->topData().resetOob();
}

void ExprDeoob::describeTo( String& str ) const
{
   str = "^-";
   str += m_first->describe();
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

void ExprXorOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   ctx->topData().xorOob();
}

void ExprXorOob::describeTo( String& str ) const
{
   str = "^%";
   str += m_first->describe();
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

void ExprIsOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   Item& item = ctx->topData();
   item.setBoolean(item.isOob());
}

void ExprIsOob::describeTo( String& str ) const
{
   str = "^?";
   str += m_first->describe();
}

}

/* end of exproob.cpp */
