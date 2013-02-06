/*
   FALCON - The Falcon Programming Language.
   FILE: stmtloop.cpp

   Statatement -- loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Feb 2013 12:49:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/stmtloop.cpp"

#include <falcon/trace.h>
#include <falcon/psteps/stmtloop.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

StmtLoop::StmtLoop(int32 line, int32 chr ):
   StmtWhile( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_loop );
   
   apply = apply_pure_;
}

StmtLoop::StmtLoop( Expression* expr, TreeStep* stmts, int32 line, int32 chr ):
   StmtWhile( expr, stmts, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_loop );

   apply = apply_pure_;
}


StmtLoop::StmtLoop( TreeStep* stmts, int32 line, int32 chr ):
         StmtWhile( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_loop );
   apply = apply_pure_;
   stmts->setParent(this);
   m_child = stmts;
}

StmtLoop::StmtLoop( Expression* expr, int32 line, int32 chr ):
         StmtWhile( expr, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_loop );
   apply = apply_withexpr_;
}


StmtLoop::StmtLoop( const StmtLoop& other ):
   StmtWhile( other )
{
   FALCON_DECLARE_SYN_CLASS( stmt_loop );
   apply = other.apply;
}

StmtLoop::~StmtLoop()
{
}


void StmtLoop::oneLinerTo( String& tgt ) const
{
   tgt = "loop ...";
}


void StmtLoop::describeTo( String& tgt, int depth ) const
{
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt += prefix + "loop\n";
   if( m_child !=0 )
   {
      tgt += m_child->describe(depth+1) + "\n";
   }
   
   if( m_expr == 0 )
   {
      tgt += prefix + "end";
   }
   else {
      tgt += prefix + "end " + m_expr->describe(depth+1);
   }
}


bool StmtLoop::selector( Expression* e )
{
   if( e == 0 )
   {
      apply = apply_pure_;
      delete m_expr;
      m_expr = 0;
      return true;
   }
   else {
      if( e->setParent(this) )
      {
         apply = apply_withexpr_;
         delete m_expr;
         m_expr = e;
         return true;
      }
   }

   return false;
}


void StmtLoop::apply_pure_( const PStep* s1, VMContext* ctx )
{
   const StmtLoop* self = static_cast<const StmtLoop*>(s1);
   TRACE( "StmtLoop::apply_pure_ entering %s", self->oneLiner().c_ize() );

   TreeStep* tree = self->m_child;
   if( tree )
   {
      CodeFrame& cf = ctx->currentCode();
      if( cf.m_seqId == 0 )
      {
         cf.m_seqId = 1;
      }
      else {
         // remove the data from the child
         ctx->popData();
      }

      ctx->stepIn(tree);
   }
   else {
      ctx->pushData(Item());
      ctx->popCode();
   }
}


void StmtLoop::apply_withexpr_( const PStep* s1, VMContext* ctx )
{
   const StmtLoop* self = static_cast<const StmtLoop*>(s1);
   TRACE( "StmtWhile::apply_withexpr_ entering %s", self->oneLiner().c_ize() );
   fassert( self->m_expr != 0 );

   CodeFrame& cf = ctx->currentCode();

   // Perform the check
   TreeStep* tree = self->m_child;
   switch ( cf.m_seqId )
   {
   case 0:
      // step in the tree
      cf.m_seqId = 1;
      if( tree != 0 ) {
         if( ctx->stepInYield( tree, cf ) )
         {
            return;
         }
      }
      /* no break */

   case 1:
      // reset status
      cf.m_seqId = 2;
      ctx->popData(); // pop the syntree result
      // generate the expression
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
         // ignore soft exception, we're yielding back soon anyhow.
         return;
      }

   /* no break */
   }

   // break items are always nil, and so, false.
   if ( ! ctx->boolTopData() )
   {
      ctx->popData();
      TRACE1( "Apply 'loop' at line %d -- redo", self->line() );
      cf.m_seqId = 1; // ... so we'll check the expression again.
      if( tree != 0 )
      {
         ctx->pushCode( tree );
      }
   }
   else {
      TRACE1( "Apply 'loop' at line %d -- leave ", self->line() );
      //we're done; recycle the top value
      ctx->topData().setNil();
      ctx->popCode();
   }
}
}

/* end of stmtloop.cpp */
