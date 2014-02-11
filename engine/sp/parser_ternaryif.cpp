/*
   FALCON - The Falcon Programming Language.
   FILE: parser_ternaryif.cpp

   Parser for Falcon source files -- ternary if expression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_ternaryif.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>

#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_ternaryif.h>

#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/exprvalue.h>

#include "falcon/synclasses_id.h"
#include "falcon/psteps/expriif.h"

namespace Falcon {

using namespace Parsing;

void apply_expr_ternary_if( const NonTerminal&, Parser& p )
{
   // << Expr << T_QMark << Expr << T_Colon << Expr
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tselector = p.getNextToken();
   p.getNextToken(); // qmark
   TokenInstance* tsecond = p.getNextToken();
   p.getNextToken(); // colon
   TokenInstance* tthird = p.getNextToken();

   ExprIIF* exprIIf = new ExprIIF( 
      static_cast<Expression*>(tselector->detachValue()), 
      static_cast<Expression*>(tsecond->detachValue()), 
      static_cast<Expression*>(tthird->detachValue()),
      tselector->line(), tselector->chr() );
   
   TokenInstance* ti = TokenInstance::alloc( tselector->line(), tselector->chr(), sp.Expr );
   ti->setValue( exprIIf, treestep_deletor );
   // clear the stack -- leave tselector
   p.simplify(5, ti);
}


}

/* end of parser_ternaryif.cpp */
