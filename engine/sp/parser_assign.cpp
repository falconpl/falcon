/*
   FALCON - The Falcon Programming Language.
   FILE: parser_assign.cpp

   Falcon source parser -- assignment handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:56:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_assign.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_assign.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprarray.h>
#include <falcon/psteps/exprassign.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

// error handler for expr_assing falls under the Expr token error handler

void apply_expr_assign( const NonTerminal&, Parser& p )
{
   // << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   Expression* firstPart = static_cast<Expression*>(v1->detachValue());
   if( firstPart->lvalueStep() == 0 )
   {
      p.addError( e_assign_sym, p.currentSource(), v1->line(), v1->chr() );
   }

   Expression* secondPart = static_cast<Expression*>(v2->detachValue());
   TokenInstance* ti = TokenInstance::alloc(v1->line(), v1->chr(), sp.Expr);
   
   // first access, then define
   ctx->accessSymbols(secondPart);
   ctx->defineSymbols(firstPart);
   ti->setValue(
      new ExprAssign( firstPart, secondPart, v1->line(), v1->chr() ),
      treestep_deletor );

   p.simplify(3,ti);
}

}

/* end of parser_assign.cpp */
