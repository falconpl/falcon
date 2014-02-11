/*
   FALCON - The Falcon Programming Language.
   FILE: parser_while.cpp

   Parser for Falcon source files -- while statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_while.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>
#include <falcon/error.h>

#include <falcon/psteps/stmtwhile.h>
#include <falcon/psteps/stmtcontinue.h>
#include <falcon/psteps/stmtbreak.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_while.h>

namespace Falcon {

using namespace Parsing;

bool while_errhand(const NonTerminal&, Parser& p, int)
{
   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = p.getLastToken();

   if( p.lastErrorLine() < ti->line() )
   {
      // generate another error only if we didn't notice it already.
      p.addError( e_syn_if, p.currentSource(), ti->line(), ti2->chr() );
   }

   if( ! p.interactive() )
   {
      // put in a fake if statement (else, subsequent else/elif/end would also cause errors).
      SynTree* stWhile = new SynTree;
      StmtWhile* fakeWhile = new StmtWhile( new ExprValue(1), stWhile, ti->line(), ti->chr() );
      ParserContext* st = static_cast<ParserContext*>(p.context());
      st->openBlock( fakeWhile, stWhile );
   }
   else
   {
      MESSAGE2( "while_errhand -- Ignoring WHILE in interactive mode." );
   }
   // on interactive parsers, let the whole instruction to be destroyed.

   p.setErrorMode(&p.T_EOF);
   return true;
}

static void apply_while_internal( Parser& p, bool autoClose )
{
   // << (r_while_short << "while_short" << apply_while_short << T_while << Expr << T_Colon

   TokenInstance* twhile = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());
   st->accessSymbols( expr );

   SynTree* whsyn = new SynTree;

   StmtWhile* stmt_wh = new StmtWhile(expr, whsyn, twhile->line(), twhile->chr());
   stmt_wh->decl( twhile->line(), twhile->chr() );
   st->openBlock( stmt_wh, whsyn, autoClose );
   
   // clear the stack
   p.simplify(3);
}

void apply_while_short( const NonTerminal&, Parser& p )
{
   apply_while_internal(p, true);
}

void apply_while( const NonTerminal&, Parser& p )
{
   apply_while_internal(p, false);
}

void apply_continue( const NonTerminal&, Parser& p )
{
   // << T_continue << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getLastToken();
   Statement* stmt = new StmtContinue( ti->line(), ti->chr() );
   ctx->addStatement( stmt );
   
   // clear the stack
   p.simplify(2);
}

void apply_break( const NonTerminal&, Parser& p )
{
   // << T_break << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getLastToken();
   Statement* stmt = new StmtBreak( ti->line(), ti->chr() );
   ctx->addStatement( stmt );
   
   // clear the stack
   p.simplify(2);   
}

}

/* end of parser_while.cpp */
