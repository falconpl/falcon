/*
   FALCON - The Falcon Programming Language.
   FILE: ExprFSelf.cpp

   Syntactic tree item definitions -- Self accessor expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/psteps/exprfself.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include "falcon/function.h"

namespace Falcon {

ExprFSelf::ExprFSelf( int line, int chr ):
   Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_fself )
   apply = apply_;
}

ExprFSelf::ExprFSelf( const ExprFSelf &other ):
   Expression(other)
{
   apply = apply_;
}

ExprFSelf::~ExprFSelf() {}


bool ExprFSelf::isStatic() const
{
   return false;
}

ExprFSelf* ExprFSelf::clone() const
{
   return new ExprFSelf( *this );
}

bool ExprFSelf::simplify( Item& ) const
{
   return false;
}

void ExprFSelf::describeTo( String & str, int ) const
{
   str = "fself";
}

void ExprFSelf::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   if( ctx->callDepth()>=1 ) 
   {
      register Function* func = ctx->currentFrame().m_function;
      ctx->pushData( Item( func->handler(), func ) );
   }
   else 
   {
      ctx->pushData(Item());
   }
}

}

/* end of ExprFSelf.cpp */
