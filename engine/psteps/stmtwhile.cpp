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

#define SRC "engine/psteps/stmtwhile.cpp"

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
   m_child( 0 ),
   m_expr( 0 )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}

StmtWhile::StmtWhile( Expression* expr, TreeStep* stmts, int32 line, int32 chr ):
   Statement( line, chr ),
   m_child( stmts ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   stmts->setParent(this);
   expr->setParent(this);
   
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}


StmtWhile::StmtWhile( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_child( 0 ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   expr->setParent(this);
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}



StmtWhile::StmtWhile( const StmtWhile& other ):
   Statement( other ),
   m_child( 0 ),
   m_expr( 0 )
{
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
   
   if( other.m_child )
   {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }
   
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);      
   }
}

StmtWhile::~StmtWhile()
{
   delete m_child;
   delete m_expr;
}


void StmtWhile::minimize()
{
   m_child = minimize_basic(m_child);
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
   if( m_expr == 0 )
   {
      tgt = "<Blank StmtWhile>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt += prefix + "while " + m_expr->describe(depth+1) + "\n";
   if( m_child !=0 )
   {
      tgt += m_child->describe(depth+1) + "\n";
   }
   tgt += prefix + "end";
}

int StmtWhile::arity() const
{
   return 1;
}

TreeStep* StmtWhile::nth( int n ) const
{
   if( n == 0 || n == -1 ) return m_child;
   return 0;
}


bool StmtWhile::setNth( int n, TreeStep* st )
{
   if( st == 0 || (n != 0 && n != -1) || ! st->setParent(this)  ) return false;

   delete m_child;
   m_child = st;

   return true;
}

void StmtWhile::mainBlock(TreeStep* st) {
   delete m_child;
   st->setParent(this);
   m_child = st;
}

TreeStep* StmtWhile::detachMainBlock()
{
   m_child->setParent(0);
   TreeStep* ret = m_child;
   m_child = 0;
   return ret;
}


void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   TRACE( "StmtWhile::apply_ entering %s", self->oneLiner().c_ize() );
   fassert( self->m_expr != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   
   // Perform the check
   TreeStep* tree = self->m_child;
   switch ( cf.m_seqId )
   {
   case 0:
      // preprare the stack
      ctx->saveUnrollPoint( cf );

      // generate the first expression
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
          // ignore soft exception, we're yielding back soon anyhow.
          return;
      }
      break;

   case 1:
      // already been around
      ctx->popData(); // remove the data placed by the syntree

      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
          // ignore soft exception, we're yielding back soon anyhow.
          return;
      }
      break;
   }
   

   // break items are always nil, and so, false.
   if ( ctx->boolTopData() )
   {
      ctx->popData();
      // mark for regeneration of the expression
      cf.m_seqId = 1;
      TRACE1( "Apply 'while' at line %d -- redo", self->line() );
      // redo
      if( tree != 0 ) {
         ctx->stepIn( tree );
      }
      else {
         return;
      }
      // no matter if stmts went deep, we're bound to be called again to recheck
   }
   else {      
      TRACE1( "Apply 'while' at line %d -- leave ", self->line() );
      //we're done
      //keep the data
      ctx->topData().setNil();
      ctx->popCode();
   }
}
         
}

/* end of stmtwhile.cpp */
