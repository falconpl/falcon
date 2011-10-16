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

   check->precompile(&m_pcCheck);
   m_pcCheck.setNextBase();

   // push ourselves and the expression in the steps
   m_step0 = this;
   m_step1 = &m_pcCheck;
}

StmtWhile::~StmtWhile()
{
   delete m_check;
   delete m_stmts;
}

void StmtWhile::oneLinerTo( String& tgt ) const
{
   tgt = "while " + m_check->describe();
}


void StmtWhile::describeTo( String& tgt ) const
{
   for( int32 i = 1; i < chr(); i++ ) {
      tgt.append(' ');
   }
   
   tgt += "while " + m_check->describe() + "\n" +
           m_stmts->describe() +
         "end\n";
}

void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   
   // break items are always nil, and so, false.
   CodeFrame& ctxTop = ctx->currentCode();
   if ( ctx->boolTopData() )
   {
      TRACE1( "Apply 'while' at line %d -- redo ", self->line() );
      // redo.
      ctx->pushCode( &self->m_pcCheck );
      ctx->pushCode( self->m_stmts );
   }
   else {
      if( &ctxTop != &ctx->currentCode() )
      {
         TRACE1( "Apply 'while' at line %d -- going deep on boolean check ", self->line() );
         return;
      }
      
      TRACE1( "Apply 'while' at line %d -- leave ", self->line() );
      //we're done
      ctx->popCode();
   }
   
   // in both cases, the data is used.
   ctx->popData();
}

}

/* end of stmtwhile.cpp */
