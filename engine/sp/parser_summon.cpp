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
#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>
#include <falcon/parser/state.h>

//#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprvalue.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

void apply_expr_summon( const Rule&, Parser& p )
{
   MESSAGE("Found SUMMON");

   SourceParser& sp = static_cast<SourceParser&>(p);

   /*
   static Engine* einst = Engine::instance();

   //   << Expr << T_DoubleColon << T_Name << T_Openpar << ListExpr << T_Closepar
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();


   List* list = static_cast<List*>(v2->detachValue());
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      // symbols will be accessed in the call expression.
      //ctx->accessSymbols(*iter);
      if( call != 0 ) call->add( *iter );
      else callps->add( *iter );
      ++iter;
   }
   // free the expressions in the list
   list->clear();

   if( call != 0 )
   {
      ti->setValue( call, treestep_deletor );
   }
   else
   {
      ti->setValue( callps, treestep_deletor );
   }
   p.simplify(3,ti);
   */
   
   TokenInstance* ti = TokenInstance::alloc(0,0, sp.Atom);
   ExprValue* v = new ExprValue(0);
   v->item(FALCON_GC_HANDLE(new String("Hello")));
   ti->setValue( v, treestep_deletor );
   p.simplify(6,ti);
}



void apply_expr_opt_summon( const Rule&, Parser& p )
{
   //  << Expr << T_COLONDOUBT << T_Name << ListExpr
   p.simplify(4);
}

}

/* end of parser_summon.cpp */
