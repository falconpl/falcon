/*
   FALCON - The Falcon Programming Language.
   FILE: stmtraise.cpp

   Syntactic tree item definitions -- raise.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 23:03:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtraise.cpp"

#include <falcon/vmcontext.h>
#include <falcon/expression.h>

#include <falcon/psteps/stmtraise.h>

namespace Falcon {

StmtRaise::StmtRaise( Expression* risen, int32 line, int32 chr ):
   Statement( Statement::e_stmt_raise, line, chr ),
   m_expr( risen )
{
   apply = apply_;
}

StmtRaise::~StmtRaise()
{
   delete m_expr;
}


void StmtRaise::describeTo( String& tgt, int depth ) const
{
   tgt = String(" ").replicate( depth * depthIndent ) +
         "raise " + m_expr->describe( depth + 1 );
   
}

void StmtRaise::oneLinerTo( String& tgt ) const
{
   tgt = "raise " + m_expr->oneLiner();
   
}

void StmtRaise::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtRaise* rs = static_cast<const StmtRaise*>( ps );
   
   CodeFrame& curCode = ctx->currentCode();
   // first time around?
   if( curCode.m_seqId == 0 )
   {
      // -- mark for second time around
      curCode.m_seqId++;
      // -- execute the expression
      ctx->stepIn( rs->m_expr );
      if( &curCode != &ctx->currentCode() )
      {
         // if went deep, try later.
         return;
      }
   }
      
   // it's pretty useless to pop things from the stack, 
   // as this operation will unroll it (code stack included) and/or throw.
   ctx->raiseItem( ctx->topData() );
}

}

/* end of stmtraise.cpp */
