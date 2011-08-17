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
#define SRC "engine/stmtraise.cpp"

#include <falcon/stmtraise.h>
#include <falcon/vmcontext.h>
#include <falcon/expression.h>

namespace Falcon {

StmtRaise::StmtRaise( Expression* risen, int32 line, int32 chr ):
   Statement( Statement::e_stmt_raise, line, chr ),
   m_expr( risen )
{
   apply = apply_;
   risen->precompile( &m_pcExpr );
   m_step0 = this;
   m_step1 = &m_pcExpr;
}

StmtRaise::~StmtRaise()
{
   delete m_expr;
}


void StmtRaise::describeTo( String& tgt ) const
{
   tgt = "raise " + m_expr->describe();
   
}


void StmtRaise::apply_( const PStep*, VMContext* ctx )
{
   // it's pretty useless to pop things from the stack, 
   // as this operation will unroll it (code stack included) and/or throw.
   ctx->raiseItem( ctx->topData() );
}

}

/* end of stmtraise.cpp */
