/*
   FALCON - The Falcon Programming Language.
   FILE: parser_end.cpp

   Parser for Falcon source files -- end statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 19:10:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_end.cpp"

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/error.h>

#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include "private_types.h"
#include <falcon/sp/parser_end.h>


namespace Falcon {

using namespace Parsing;

void apply_end( const NonTerminal&, Parser& p )
{
   // << T_end  << T_EOL

   ParserContext* st = static_cast<ParserContext*>(p.context());
   //Statement* current = st->currentStmt();
   if( !st->currentStmt() && !st->currentFunc() && !st->currentClass())
   {
      // clear the stack
      TokenInstance* tend = p.getNextToken();
      p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
      p.simplify(2);
      return;
   }

   // clear the stack
   p.simplify(2);
   st->closeContext();
}

void apply_end_small( const NonTerminal&, Parser& p )
{
   // << apply_end << T_end  )

   ParserContext* st = static_cast<ParserContext*>(p.context());
   //Statement* current = st->currentStmt();
   if( !st->currentStmt() && !st->currentFunc() && !st->currentClass())
   {
      TokenInstance* tend = p.getNextToken();
      p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
      p.simplify(1);
      p.consumeUpTo( p.T_EOL );      
      return;
   }

   // clear the stack
   p.simplify(1);
   st->closeContext();
}


void apply_end_rich( const NonTerminal&, Parser& p )
{
   // << apply_end_rich << T_end << Expr << T_EOL )
   TokenInstance* tend = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   // TODO: Actually, it's used for the Loop statement.
   if( !st->currentStmt() )
   {
      delete expr;
      p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
   }
   else
   {
      TreeStep* statement = st->currentStmt();
      if( ! statement->selector( expr ) )
      {
         delete expr;
         p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
      }
   }

   //SourceParser* sp = static_cast<SourceParser*>( &p);
   if( !p.interactive() && (st->currentStmt() || st->currentFunc() || st->currentClass()) )
   {
      // close the current context even in case of error.
      st->closeContext();
   }
   // clear the stack
   p.simplify(3);
}

}

/* end of parser_end.cpp */
