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
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_rule.h>

#include <falcon/psteps/exprrule.h>

#include "private_types.h"
#include "falcon/synclasses_id.h"


namespace Falcon {

using namespace Parsing;


void apply_rule( const NonTerminal&, Parser& p )
{
   // << (r_rule << "rule" << apply_rule << T_rule << T_EOL )
   TokenInstance* trule = p.getNextToken();
   SourceParser* sp = static_cast<SourceParser*>(&p);

   ParserContext* st = static_cast<ParserContext*>(p.context());
   ExprRule* stmt_rule = new ExprRule( trule->line(), trule->chr() );

   // clear the stack
   p.trimFromBase(1,1);

   // our rule is now an expression
   trule->token( sp->Expr );
   trule->setValue( stmt_rule, treestep_deletor );

   // enter the rule.
   st->openBlock( stmt_rule, &stmt_rule->currentTree(), false, false );
   p.pushState( "InlineFunc" );
}

void apply_rule_branch( const NonTerminal&, Parser& p )
{
   // << apply_rule_branch << T_or << T_EOL )
   TokenInstance* trule = p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());
   ExprRule* stmt_rule = static_cast<ExprRule*>(st->currentStmt());
   if( stmt_rule == 0 || stmt_rule->handler()->userFlags() != FALCON_SYNCLASS_ID_RULE )
   {
      p.addError( e_syn_or, p.currentSource(), trule->line(), trule->chr() );
   }
   else {
      stmt_rule->addAlternative();
      SynTree* tree = &stmt_rule->currentTree();
      st->changeBranch( tree );
   }

   // clear the stack
   p.simplify(2);
}

static void apply_cut_internal( const NonTerminal&, Parser& p, bool hasExpr )
{
   TokenInstance* trule = p.getNextToken();
   Expression* expr = 0;
   if( hasExpr )
   {
      expr = static_cast<Expression*>(p.getNextToken()->detachValue());
   }

   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || 
      ( st->currentStmt()->handler()->userFlags() != FALCON_SYNCLASS_ID_RULE) )
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


void apply_cut_expr( const NonTerminal& r, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << Expr << T_EOL )
   apply_cut_internal( r, p, true );
}


void apply_cut( const NonTerminal& r, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << T_EOL )
   apply_cut_internal( r, p, false );
}


void apply_doubt( const NonTerminal&, Parser& p )
{
   TokenInstance* trule = p.getNextToken();
   Expression* expr = static_cast<Expression*>(p.getNextToken()->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || 
      ( st->currentStmt()->handler()->userFlags() != FALCON_SYNCLASS_ID_RULE) )
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
