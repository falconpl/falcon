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
#include <falcon/psteps/exprpseudocall.h>
#include <falcon/psteps/exprsym.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

void apply_expr_call( const Rule&, Parser& p )
{
   static Engine* einst = Engine::instance();

   // r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   // Our call expression
   ExprCall* call = 0;
   ExprPseudoCall *callps = 0;

   TokenInstance* ti = TokenInstance::alloc(v1->line(), v1->chr(), sp.Expr);
   Expression* callee = static_cast<Expression*>(v1->detachValue());
   ctx->accessSymbols(callee);
   if( callee->trait() == Expression::e_trait_symbol )
   {
      // check if the symbol is a pseudofunction   
      ExprSymbol* esym = static_cast<ExprSymbol*>(callee);
      PseudoFunction* pf = static_cast<PseudoFunction*>(
            einst->getMantra( esym->name(), Mantra::e_c_pseudofunction ));

      // if it is, we don't need the callee expression anymore.
      if( pf != 0 )
      {
         callps = new ExprPseudoCall(pf, v1->line(), v1->chr());
         delete esym;
      }
      else
      {
         call = new ExprCall( callee, v1->line(), v1->chr() );
      }
   }
   else {
      call = new ExprCall( callee, v1->line(), v1->chr() );
   }

   List* list = static_cast<List*>(v2->detachValue());
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      ctx->accessSymbols(*iter);
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
   
   p.simplify(4,ti);
}

}

/* end of parser_call.cpp */
