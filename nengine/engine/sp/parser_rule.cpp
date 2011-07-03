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

void apply_cut( const Rule&, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << T_EOL )
   TokenInstance* trule = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || st->currentStmt()->type() != Statement::rule_t )
   {
      p.addError( e_syn_cut, p.currentSource(), trule->line(), trule->chr() );
   }
   else
   {
      StmtCut* stmt_cut = new StmtCut( trule->line(), trule->chr() );
      static_cast<StmtRule*>(st->currentStmt())->addStatement(stmt_cut);
   }

   // clear the stack
   p.simplify(2);
}

}

/* end of parser_rule.cpp */
