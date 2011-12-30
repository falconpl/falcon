/*
   FALCON - The Falcon Programming Language.
   FILE: exprself.cpp

   Syntactic tree item definitions -- Self accessor expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/psteps/exprself.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

ExprSelf::ExprSelf( int line, int chr ):
   Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_self )
   apply = apply_;
}

ExprSelf::ExprSelf( const ExprSelf &other ):
   Expression(other)
{
   apply = apply_;
}

ExprSelf::~ExprSelf() {}


bool ExprSelf::isStatic() const
{
   return false;
}

ExprSelf* ExprSelf::clone() const
{
   return new ExprSelf( *this );
}

bool ExprSelf::simplify( Item& ) const
{
   return false;
}

void ExprSelf::describeTo( String & str, int ) const
{
   str = "self";
}

void ExprSelf::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->pushData(ctx->currentFrame().m_self);
}

}

/* end of exprself.cpp */
