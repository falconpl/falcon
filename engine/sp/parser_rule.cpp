/*
   FALCON - The Falcon Programming Language.
   FILE: parser_rule.cpp

   Parser for Falcon source files -- rule declaration handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_rule.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/stmtrule.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include "private_types.h"
#include <falcon/sp/parser_rule.h>


namespace Falcon {

using namespace Parsing;

void apply_rule( const Rule&, Parser& p )
{
   // << (r_rule << "rule" << apply_rule << T_rule << T_EOL )
   TokenInstance* trule = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   StmtRule* stmt_rule = new StmtRule( trule->line(), trule->chr() );
   st->openBlock( stmt_rule, &stmt_rule->currentTree() );

   // clear the stack
   p.simplify(2);
}



static void apply_cut_internal( const Rule&, Parser& p, bool hasExpr )
{
   TokenInstance* trule = p.getNextToken();
   Expression* expr = 0;
   if( hasExpr )
   {
      expr = static_cast<Expression*>(p.getNextToken()->detachValue());
   }

   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || st->currentStmt()->type() != Statement::e_stmt_rule )
   {
      p.addError( e_syn_cut, p.currentSource(), trule->line(), trule->chr() );
   }
   else
   {
      StmtCut* stmt_cut = new StmtCut( expr, trule->line(), trule->chr() );
      st->addStatement( stmt_cut );
   }

   // clear the stack
   p.simplify( expr != 0 ? 3 : 2);
}


void apply_cut_expr( const Rule& r, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << Expr << T_EOL )
   apply_cut_internal( r, p, true );
}


void apply_cut( const Rule& r, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << T_EOL )
   apply_cut_internal( r, p, false );
}


void apply_doubt( const Rule&, Parser& p )
{
   TokenInstance* trule = p.getNextToken();
   Expression* expr = static_cast<Expression*>(p.getNextToken()->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || st->currentStmt()->type() != Statement::e_stmt_rule )
   {
      p.addError( e_syn_doubt, p.currentSource(), trule->line(), trule->chr() );
   }
   else
   {
      StmtDoubt* stmt_doubt = new StmtDoubt( expr, trule->line(), trule->chr() );
      st->addStatement( stmt_doubt );
   }

   // clear the stack
   p.simplify( 3 );
}

}

/* end of parser_rule.cpp */
