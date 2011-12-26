/*
   FALCON - The Falcon Programming Language.
   FILE: stmtwhile.cpp

   Statatement -- while
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/stmtwhile.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>

namespace Falcon
{

StmtWhile::StmtWhile( Expression* check, SynTree* stmts, int32 line, int32 chr ):
   Statement( e_stmt_while, line, chr ),
   m_check(check),
   m_stmts( stmts )
{
   apply = apply_;
   m_bIsLoopBase = true;
}

StmtWhile::~StmtWhile()
{
   delete m_check;
   delete m_stmts;
}

void StmtWhile::oneLinerTo( String& tgt ) const
{
   tgt = "while " + m_check->oneLiner();
}


void StmtWhile::describeTo( String& tgt, int depth ) const
{
   String prefix = String(" ").replicate( depth * depthIndent );
   
   tgt += prefix + "while " + m_check->describe(depth+1) + "\n" +
           m_stmts->describe(depth+1) + "\n" +
         prefix + "end";
}

void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   
   CodeFrame& cf = ctx->currentCode();
   
   // Perform the check
   if ( cf.m_seqId == 0 ) 
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_check, cf ) )
      {
         // ignore soft exception, we're yielding back soon anyhow.
         return;
      }
   }
   // otherwise, we're here after performing a check.
   cf.m_seqId = 0; // ... so we set it back to perform check.
   
   // break items are always nil, and so, false.
   if ( ctx->boolTopData() )
   {
      ctx->popData();
      TRACE1( "Apply 'while' at line %d -- redo", self->line() );
      // redo
      ctx->stepIn( self->m_stmts );
      // no matter if stmts went deep, we're bound to be called again to recheck
   }
   else {      
      TRACE1( "Apply 'while' at line %d -- leave ", self->line() );
      //we're done
      ctx->popData();
      ctx->popCode();
   }
}
         
}

/* end of stmtwhile.cpp */
