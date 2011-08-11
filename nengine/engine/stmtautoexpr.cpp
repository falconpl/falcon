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
#define SRC "engine/stmtautoexpr.cpp"

#include <falcon/stmtautoexpr.h>
#include <falcon/expression.h>
#include <falcon/syntree.h>
#include <falcon/vmcontext.h>

#include <falcon/codeerror.h>
#include <falcon/trace.h>

namespace Falcon 
{

StmtAutoexpr::StmtAutoexpr( Expression* expr, int32 line, int32 chr ):
      Statement(e_stmt_autoexpr, line, chr ),
      m_expr( expr ),
      m_nd( false ),
      m_determ( false )
{
   apply = apply_;
   
   m_expr->precompile(&m_pcExpr);

   // Push ourselves
   m_step0 = this;
   m_step1 = &m_pcExpr;
}

StmtAutoexpr::~StmtAutoexpr()
{
   delete m_expr;
}

void StmtAutoexpr::describeTo( String& tgt ) const
{
   for( int32 i = 1; i < chr(); i++ ) {
      tgt.append(' ');
   }

   if( m_nd ) tgt += "? ";
   else if( m_determ ) tgt += "* ";
   
   tgt += m_expr->describe();
}

void StmtAutoexpr::oneLinerTo( String& tgt ) const
{
   tgt += m_expr->describe();
}

void StmtAutoexpr::nd( bool mode )
{
   if ( m_determ )
      throw new CodeError( ErrorParam( e_determ_decl, __LINE__, __FILE__ ).extra( "setting nd" ) );

   m_nd = mode;
}

void StmtAutoexpr::determ( bool mode )
{
   if ( m_nd )
      throw new CodeError( ErrorParam( e_determ_decl, __LINE__, __FILE__ ).extra( "setting determ" ) );

   m_determ = mode;
}

void StmtAutoexpr::apply_( const PStep* self, VMContext* ctx )
{
   TRACE3( "StmtAutoexpr apply: %p (%s)", self, self->describe().c_ize() );

   // we never need to be called again.
   ctx->popCode();
   
   // save the result in the A register
   ctx->regA() = ctx->topData();

   // remove the data created by the expression
   ctx->popData();

   // finally, check determinism.
   const StmtAutoexpr* sae = static_cast<const StmtAutoexpr*>(self);
   if( sae->m_determ ) 
   {
      ctx->checkNDContext();
   }
   else if( sae->m_nd ) 
   {
      ctx->SetNDContext();
   }
}

}

/* end of stmtautoexpr.cpp */
