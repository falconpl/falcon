/*
   FALCON - The Falcon Programming Language.
   FILE: stmtreturn.cpp

   Statatement -- return
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/stmtreturn.h>

namespace Falcon
{

StmtReturn::StmtReturn( Expression* expr, int32 line, int32 chr ):
   Statement(e_stmt_return, line, chr ),
   m_expr( expr ),
   m_bHasDoubt( false )
{
   m_step0 = this;

   if ( expr )
   {
      m_expr = expr;
      expr->precompile( &m_pcExpr );
      m_step1 = &m_pcExpr;
      apply = apply_expr_;
   }
   else
   {
      apply = apply_;
   }
}

StmtReturn::~StmtReturn()
{
   delete m_expr;
}

void StmtReturn::expression( Expression* expr )
{
   delete m_expr;
   m_expr = expr;
   apply = m_bHasDoubt ? apply_expr_doubt_ : apply_expr_;
}


void StmtReturn::hasDoubt( bool b )
{
   m_bHasDoubt = b; 
   if( b )
   {
      apply = m_expr == 0 ? apply_expr_doubt_ : apply_doubt_;
   }
   else
   {
      apply = m_expr == 0 ? apply_expr_ : apply_;
   }
}
 

void StmtReturn::describeTo( String& tgt ) const
{
   tgt = "return";
   
   if( m_bHasDoubt )
   {
      tgt += " ?";
   }
   
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->describe();
   }   
}


void StmtReturn::apply_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return'" );   
   ctx->returnFrame();
}


void StmtReturn::apply_expr_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return expr'" );
   ctx->returnFrame( ctx->topData() );
}

void StmtReturn::apply_doubt_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return ?'");   
   ctx->returnFrame();
   ctx->SetNDContext();
}


void StmtReturn::apply_expr_doubt_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return expr'" );
   ctx->returnFrame( ctx->topData() );
   ctx->SetNDContext();
}

}

/* end of stmtreturn.cpp */
