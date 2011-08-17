/*
   FALCON - The Falcon Programming Language.
   FILE: parser_assign.cpp

   Falcon source parser -- assignment handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:56:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_assign.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_assign.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/expression.h>
#include <falcon/exprarray.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

// error handler for expr_assing falls under the Expr token error handler

void apply_expr_assign( const Rule&, Parser& p )
{
   // << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << NeListExpr)
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   Expression* firstPart = static_cast<Expression*>(v1->detachValue());

   // assignable expressions are only expressions having a lvalue pstep:
   // -- symbols
   // -- accessors
   if( firstPart->lvalueStep() == 0  )
   {
      p.addError( e_assign_sym, p.currentSource(), v1->line(), v1->chr(), 0 );
   }
   else
   {
      ctx->defineSymbols(firstPart);
   }
   
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);

   // do not detach, we don't care about the list
   List* list = static_cast<List*>(v2->asData());
   fassert( ! list->empty() );
   if( list->size() == 1 )
   {
       ti->setValue(
         new ExprAssign( firstPart, list->front() ),
         expr_deletor );
   }
   else
   {
      // a list assignment.

      ExprArray* array = new ExprArray;
      List::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         array->add(*iter);
         ++iter;
      }

      ti->setValue(
         new ExprAssign( firstPart, array ),
         expr_deletor );
   }
   // clear, so we keep the expr even if destroyed
   list->clear();

   p.simplify(3,ti);
}

#if 0
//TODO Remove
static void apply_expr_list( const Rule&, Parser& p )
{
   //<< (r_Expr_list << "Expr_list" << apply_expr_list << ListExpr )
   parser_assign& sp = static_cast<parser_assign&>(p);

   TokenInstance* v1 = p.getNextToken();

   List* list = static_cast<List*>(v1->detachValue());

   ExprArray* array = new ExprArray;
   // it's a dictionary declaration
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      // Todo -- make the array expression
      array->add(*iter);
      ++iter;
   }
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( array, expr_deletor );

   // free the expressions in the list
   list->clear();

   p.simplify(1,ti);
}
#endif

}

/* end of parser_assign.cpp */
