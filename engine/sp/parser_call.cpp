/*
   FALCON - The Falcon Programming Language.
   FILE: parser_call.cpp

   Falcon source parser -- function call handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:51:27 +0200
 
   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_class.cpp"

#include <falcon/setup.h>
#include <falcon/pseudofunc.h>
#include <falcon/symbol.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_call.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/nonterminal.h>
#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>
#include <falcon/parser/state.h>

#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprsym.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

void apply_expr_call( const Rule&, Parser& p )
{
   static Engine* einst = Engine::instance();

   // r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   // Our call expression
   ExprCall* call;

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   Expression* callee = static_cast<Expression*>(v1->detachValue());
   if( callee->trait() == Expression::e_trait_symbol )
   {
      // check if the symbol is a pseudofunction   
      ExprSymbol* esym = static_cast<ExprSymbol*>(callee);
      PseudoFunction* pf = einst->getPseudoFunction( esym->name() );

      // if it is, we don't need the callee expression anymore.
      if( pf != 0 )
      {
         call = new ExprCall(pf);
         ParserContext* ctx = static_cast<ParserContext*>(p.context());
         ctx->undoVariable( esym->name() );
         delete esym;
      }
      else
      {
         call = new ExprCall( callee );
      }
   }
   else {
      call = new ExprCall( callee );
   }

   List* list = static_cast<List*>(v2->detachValue());
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      call->add( *iter );
      ++iter;
   }
   // free the expressions in the list
   list->clear();

   ti->setValue( call, expr_deletor );
   p.simplify(4,ti);
}

}

/* end of parser_call.cpp */
