/*
   FALCON - The Falcon Programming Language.
   FILE: stmtautoexpr.cpp

   Syntactic tree item definitions -- Autoexpression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtautoexpr.cpp"

#include <falcon/expression.h>
#include <falcon/syntree.h>
#include <falcon/vmcontext.h>

#include <falcon/errors/codeerror.h>
#include <falcon/trace.h>

#include <falcon/psteps/stmtautoexpr.h>

namespace Falcon 
{

StmtAutoexpr::StmtAutoexpr( Expression* expr, int32 line, int32 chr ):
   Statement(e_stmt_autoexpr, line, chr ),
   m_expr( expr ),
   m_bInteractive( false ),
   m_bInRule( false )
{
   apply = apply_; 
}

StmtAutoexpr::~StmtAutoexpr()
{
   delete m_expr;
}

void StmtAutoexpr::describeTo( String& tgt, int depth ) const
{
   tgt += String(" ").replicate( depth * depthIndent ) + m_expr->describe();
}

void StmtAutoexpr::oneLinerTo( String& tgt ) const
{
   tgt += m_expr->describe();
}


void StmtAutoexpr::setInteractive( bool bInter )
{
   m_bInteractive = bInter;
   if( bInter )
   {
      apply = apply_interactive_;
   }
   else
   {
      if( m_bInRule )
      {
         apply = apply_rule_;
      }
      else
      {
         apply = apply_;
      }
   }
}


void StmtAutoexpr::setInRule( bool bMode )
{
   m_bInRule = bMode;
   if( bMode )
   {
      apply = apply_rule_;      
   }
   else
   {
      if( m_bInteractive )
      {
         apply = apply_interactive_;
      }
      else
      {
         apply = apply_;
      }
   }
}

void StmtAutoexpr::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtAutoexpr* self = static_cast<const StmtAutoexpr*>( ps );
   TRACE3( "StmtAutoexpr apply: %p (%s)", self, self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   // first time, try to run the expression.
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
     
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
         return;
      }
   }
   // just cleanup our mess.
   ctx->popData();
   ctx->popCode();
}


void StmtAutoexpr::apply_interactive_( const PStep* ps, VMContext* ctx )
{
   const StmtAutoexpr* self = static_cast<const StmtAutoexpr*>( ps );
   TRACE3( "StmtAutoexpr apply interactive: %p (%s)", self, self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   // first time, try to run the expression.
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
     
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
         return;
      }
   }
   
   // we never need to be called again.
   ctx->popCode();
   
   ctx->regA() = ctx->topData();
   
   // remove the data created by the expression
   ctx->popData();
}


void StmtAutoexpr::apply_rule_( const PStep* ps, VMContext* ctx )
{
   const StmtAutoexpr* self = static_cast<const StmtAutoexpr*>( ps );
   TRACE3( "StmtAutoexpr apply rule: %p (%s)", self, self->describe().c_ize() );
   
   CodeFrame& cf = ctx->currentCode();
   // first time, try to run the expression.
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
     
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
         return;
      }
   }

   // we never need to be called again.
   ctx->popCode();

   // we're inside a rule, or we wouldn't be called.
   register Item* td = ctx->topData().dereference();
   
   // set to false only if the last op result was a boolean false.
   ctx->ruleEntryResult( !(td->isBoolean() && td->asBoolean() == false) );
   
   // remove the data created by the expression
   ctx->popData();
}

}

/* end of stmtautoexpr.cpp */
