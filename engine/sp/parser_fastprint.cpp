/*
   FALCON - The Falcon Programming Language.
   FILE: parser_fastprint.cpp

   Parser for Falcon source files -- fast-print statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Aug 2011 12:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parsercontext.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>
#include <falcon/error.h>

#include <falcon/expression.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_fastprint.h>

#include <falcon/psteps/stmtfastprint.h>
#include <falcon/psteps/stmtfastprintnl.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;


static void apply_fastprint_internal( const NonTerminal&, Parser& p, bool hasNl )
{
   // << T_Greater << ListExpr << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   /* TokenInstance* ttok = */ p.getNextToken();
   TokenInstance* tlist = p.getNextToken();
   List* list = static_cast<List*>(tlist->asData());
   
   // create our fast-print statement
   StmtFastPrint* sfp;
   if( hasNl )
   {
      sfp = new StmtFastPrintNL( tlist->line(), tlist->chr());
   }
   else {
      sfp =  new StmtFastPrint( tlist->line(), tlist->chr());
   }
   
   // move the expressions in the statement.
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      Expression* expr = *iter;
      ctx->accessSymbols( expr );
      sfp->add( *iter );
      ++iter;
   }
   // release the expressions.
   list->clear();
   
   // we can add the statement directly
   ctx->addStatement( sfp );
   
   // clear the stack
   p.simplify(3);
}


void apply_fastprint( const NonTerminal& r, Parser& p )
{
   apply_fastprint_internal(r, p, false);
}


void apply_fastprint_nl( const NonTerminal& r, Parser& p )
{
   apply_fastprint_internal(r, p, true);
}

void apply_fastprint_alone( const NonTerminal&, Parser& p )
{
   // << T_RShift << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* tlist = p.getNextToken();
   StmtFastPrint* sfp = new StmtFastPrint( tlist->line(), tlist->chr());
   // we can add the statement directly
   ctx->addStatement( sfp );

   // clear the stack
   p.simplify(2);
}


void apply_fastprint_nl_alone( const NonTerminal&, Parser& p )
{
   // << T_Greater << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* tlist = p.getNextToken();
   StmtFastPrint* sfp = new StmtFastPrintNL( tlist->line(), tlist->chr());
   // we can add the statement directly
   ctx->addStatement( sfp );

   // clear the stack
   p.simplify(2);
}

}

/* end of parser_fastprint.cpp */
