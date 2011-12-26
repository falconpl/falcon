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

namespace Falcon
{


StmtBreak::StmtBreak( int32 line, int32 chr ):
   Statement( e_stmt_break, line, chr)
{
   apply = apply_;
}

   
void StmtBreak::describeTo( String& tgt, int depth ) const
{
   tgt = String(" ").replicate( depth * depthIndent ) + "break";
}


void StmtBreak::oneLinerTo( String& tgt, int depth ) const
{
   tgt = "break";
}


void StmtBreak::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollToLoopBase(); // which will pop us as well.
   Item b;
   b.setBreak();
   ctx->pushData( b );
}

}

/* end of stmtbreak.cpp */
