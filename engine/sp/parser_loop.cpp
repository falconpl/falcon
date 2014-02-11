/*
   FALCON - The Falcon Programming Language.
   FILE: parser_loop.cpp

   Parser for Falcon source files -- while statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Feb 2013 12:43:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_loop.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>
#include <falcon/error.h>

#include <falcon/psteps/stmtloop.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_loop.h>

namespace Falcon {

using namespace Parsing;

bool loop_errhand(const NonTerminal&, Parser& p, int)
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
      StmtWhile* fakeWhile = new StmtLoop( stWhile, ti->line(), ti->chr() );
      ParserContext* st = static_cast<ParserContext*>(p.context());
      st->openBlock( fakeWhile, stWhile );
   }
   else
   {
      MESSAGE2( "loop_errhand -- Ignoring WHILE in interactive mode." );
   }
   // on interactive parsers, let the whole instruction to be destroyed.

   p.setErrorMode(&p.T_EOF);
   return true;
}

static void apply_loop_internal( Parser& p, bool autoClose )
{
   // << (r_while_short << "while_short" << apply_while_short << T_loop << T_Colon
   TokenInstance* tloop = p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());
   SynTree* whsyn = new SynTree;

   StmtWhile* stmt_wh = new StmtLoop( whsyn, tloop->line(), tloop->chr() );
   st->openBlock( stmt_wh, whsyn, autoClose );
   
   // clear the stack
   p.simplify(2);
}

void apply_loop_short( const NonTerminal&, Parser& p )
{
   // << T_loop << T_Colon
   apply_loop_internal(p, true);
}


void apply_loop( const NonTerminal&, Parser& p )
{
   // << apply_loop << T_loop << T_EOL
   apply_loop_internal(p, false);
}

}

/* end of parser_loop.cpp */
