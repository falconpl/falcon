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
   m_postCheck( this ),
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
   
   // push a post-check step
   ctx->pushCode( &self->m_postCheck );
   CodeFrame& postCheckFrame = ctx->currentCode();
   
   // and perform the check
   ctx->stepIn( self->m_check );
   if( &ctx->currentCode() != &postCheckFrame )
   {
      // ignore soft exception, we're yielding back soon anyhow.
      return;
   }
   // we don't need the postCheck code anymore.
   ctx->popCode();
   
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



StmtWhile::PostCheck::~PostCheck()
{   
}

void StmtWhile::PostCheck::describeTo( String& tgt )
{
   m_owner->describeTo(tgt);
   tgt += " [postcheck]";
}

void StmtWhile::PostCheck::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtWhile::PostCheck* self = static_cast<const StmtWhile*>(ps);
   
   // break items are always nil, and so, false.
   if ( ctx->boolTopDataAndPop() )
   {
      TRACE1( "Apply 'while (post check)' at line %d -- redo", self->m_owner->line() );
      // redo -- avoiding to remove ourself so we can use our own pstep.
      ctx->resetAndApply( self->m_owner->m_stmts );
      // no matter if stmts went deep, we're bound to be called again to recheck
   }
   else {      
      TRACE1( "Apply 'while (post check)' at line %d -- leave ", self->line() );
      //we're done
      ctx->popCode();
   }
}
         
}

/* end of stmtwhile.cpp */
