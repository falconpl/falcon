/*
   FALCON - The Falcon Programming Language.
   FILE: stmtcontinue.cpp

   Statatement -- contunue (loop restart)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/stmtcontinue.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon
{


StmtContinue::StmtContinue( int32 line, int32 chr ):
   Statement( line, chr)
{
   FALCON_DECLARE_SYN_CLASS(stmt_continue)
   apply = apply_;
}

StmtContinue::StmtContinue( const StmtContinue& other ):
   Statement( other )
{
   apply = apply_;
}
   
void StmtContinue::describeTo( String& tgt, int depth ) const
{
   tgt = String(" ").replicate( depth * depthIndent ) + "continue";
}


void StmtContinue::oneLinerTo( String& tgt ) const
{
   tgt = "continue";
}


void StmtContinue::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollToNextBase(); // that will pop me as well.
}

}

/* end of stmtcontinue.cpp */
