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

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

StmtWhile::StmtWhile(int32 line, int32 chr ):
   Statement( line, chr ),
   m_stmts( 0 ),
   m_expr( 0 )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   apply = apply_;
   m_bIsLoopBase = true;
}

StmtWhile::StmtWhile( Expression* expr, SynTree* stmts, int32 line, int32 chr ):
   Statement( line, chr ),
   m_stmts( stmts ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   stmts->setParent(this);
   expr->setParent(this);
   
   apply = apply_;
   m_bIsLoopBase = true;
}


StmtWhile::StmtWhile( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_stmts( 0 ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   m_stmts = new SynTree;
   m_stmts->setParent(this);
   expr->setParent(this);
   apply = apply_;
   m_bIsLoopBase = true;
}



StmtWhile::StmtWhile( const StmtWhile& other ):
   Statement( other ),
   m_stmts( 0 ),
   m_expr( 0 )
{
   apply = apply_;
   m_bIsLoopBase = true;
   
   if( other.m_stmts )
   {
      m_stmts = other.m_stmts->clone();
      m_stmts->setParent(this);
   }
   
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);      
   }
}

StmtWhile::~StmtWhile()
{
   delete m_stmts;
   delete m_expr;
}


Expression* StmtWhile::selector() const
{
   return m_expr;
}


bool StmtWhile::selector( Expression* e )
{
   if( e!= 0 && e->setParent(this))
   {
      delete m_expr;
      m_expr = e;
      return true;
   }
   return false;
}
   
void StmtWhile::oneLinerTo( String& tgt ) const
{
   if( m_expr == 0 )
   {
      tgt = "<Blank StmtWhile>";
      return;
   }
   
   tgt = "while " + m_expr->oneLiner();
}


void StmtWhile::describeTo( String& tgt, int depth ) const
{
   if( m_expr == 0 || m_stmts == 0 )
   {
      tgt = "<Blank StmtWhile>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt += prefix + "while " + m_expr->describe(depth+1) + "\n" +
           m_stmts->describe(depth+1) + "\n" +
         prefix + "end";
}

int StmtWhile::arity() const
{
   return 1;
}

TreeStep* StmtWhile::nth( int n ) const
{
   if( n == 0 || n == -1 ) return m_stmts;
   return 0;
}


bool StmtWhile::setNth( int n, TreeStep* st )
{
   if( st == 0 || st->category() != TreeStep::e_cat_syntree || ! st->setParent(this) ) return false;
   
   if( n == 0 || n == -1 ) 
   {
      delete m_stmts;
      m_stmts = static_cast<SynTree*>(st);
   }
   
   return 0;
}

void StmtWhile::mainBlock(SynTree* st) {
   delete m_stmts;
   st->setParent(this);
   m_stmts = st;
}

SynTree* StmtWhile::detachMainBlock()
{
   m_stmts->setParent(0);
   SynTree* ret = m_stmts;
   m_stmts = 0;
   return ret;
}


void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   TRACE( "StmtWhile::apply_ entering %s", self->oneLiner().c_ize() );
   fassert( self->m_expr != 0 );
   fassert( self->m_stmts != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   
   // Perform the check
   SynTree* tree = self->m_stmts;
   if ( cf.m_seqId == 0 ) 
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_expr, cf ) )
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
      ctx->stepIn( tree );
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
