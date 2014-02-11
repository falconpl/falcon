/*
   FALCON - The Falcon Programming Language.
   FILE: parser_summon.cpp

   Falcon source parser -- summon expression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 17 Apr 2013 17:20:03 +0200
 
   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_summon.cpp"

#include <falcon/setup.h>
#include <falcon/pseudofunc.h>
#include <falcon/symbol.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_summon.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/nonterminal.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprsummon.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

static void internal_summon(Parser& p, bool isOptional)
{
   //  << Expr << T_ColonDoubt << T_Name << T_Openpar << ListExpr << T_Closepar
   TokenInstance* tsel = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();
   p.getNextToken();
   TokenInstance* tlist = p.getNextToken();

   ExprSummonBase* es;
   if( isOptional )
   {
      es = new ExprOptSummon(tsel->line(), tsel->chr());
   }
   else {
      es = new ExprSummon(tsel->line(), tsel->chr());
   }

   es->selector( static_cast<Expression*>(tsel->detachValue()) );
   es->message(*tname->asString());
   List* list = static_cast<List*>(tlist->asData());
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      es->append( *iter );
      ++iter;
   }
   // free the expressions in the list
   list->clear();
   // use the first token.
   tsel->setValue( es, treestep_deletor );
   p.trim(5);
}

void apply_expr_summon( const NonTerminal&, Parser& p )
{
   MESSAGE("Found SUMMON");

   //  << Expr << T_DoubleColon << T_Name << T_Openpar << ListExpr << T_Closepar
   internal_summon(p,false);
}



void apply_expr_opt_summon( const NonTerminal&, Parser& p )
{
   //  << Expr << T_ColonDoubt << T_Name << T_Openpar << ListExpr << T_Closepar
   internal_summon(p,true);
}

}

/* end of parser_summon.cpp */
