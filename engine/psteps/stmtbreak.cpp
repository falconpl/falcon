/*
   FALCON - The Falcon Programming Language.
   FILE: stmtbreak.cpp

   Statatement -- break (loop breaking)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/stmtbreak.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon
{

StmtBreak::StmtBreak( int32 line, int32 chr ):
   Statement(line, chr)
{
   FALCON_DECLARE_SYN_CLASS(stmt_break)
   apply = apply_;
}

StmtBreak::StmtBreak( const StmtBreak& other ):
   Statement(other)
{
   apply = apply_;
}

void StmtBreak::describeTo( String& tgt, int depth ) const
{
   tgt = String(" ").replicate( depth * depthIndent ) + "break";
}


void StmtBreak::oneLinerTo( String& tgt ) const
{
   tgt = "break";
}


void StmtBreak::apply_( const PStep*, VMContext* ctx )
{
   // this will throw if not found
   ctx->unrollToLoopBase();
}

}

/* end of stmtbreak.cpp */
