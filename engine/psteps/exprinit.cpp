/*
   FALCON - The Falcon Programming Language.
   FILE: exprinit.cpp

   Syntactic tree item definitions -- Init values for generators
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Feb 2013 18:11:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/psteps/exprinit.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include "falcon/function.h"

namespace Falcon {

ExprInit::ExprInit( int line, int chr ):
   Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_init )
   m_pstep_lvalue = &m_pslv;
   apply = apply_;
}

ExprInit::ExprInit( const ExprInit &other ):
   Expression(other)
{
   m_pstep_lvalue = &m_pslv;
   apply = apply_;
}

ExprInit::~ExprInit() {}


bool ExprInit::isStatic() const
{
   return false;
}


void ExprInit::render( TextWriter* tw, int ) const
{
   tw->write("init");
}


void ExprInit::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->pushData( ctx->readInit() );
}


void ExprInit::PStepLValue::describeTo( String& str ) const
{
   str = "init";
}


void ExprInit::PStepLValue::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->topData().copied(true);
   ctx->writeInit(ctx->topData());
}

}

/* end of exprinit.h */
