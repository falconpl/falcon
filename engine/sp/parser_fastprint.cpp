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
#include <falcon/stmtfastprint.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_fastprint.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;


static void apply_fastprint_internal( const Rule&, Parser& p, bool hasNl )
{
   // << T_Greater << ListExpr << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   /* TokenInstance* ttok = */ p.getNextToken();
   TokenInstance* tlist = p.getNextToken();
   List* list = static_cast<List*>(tlist->asData());
   
   // create our fast-print statement
   StmtFastPrint* sfp = new StmtFastPrint(hasNl);
   
   // move the expressions in the statement.
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
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


void apply_fastprint( const Rule& r, Parser& p )
{
   apply_fastprint_internal(r, p, false);
}


void apply_fastprint_nl( const Rule& r, Parser& p )
{
   apply_fastprint_internal(r, p, true);
}

}

/* end of parser_fastprint.cpp */
