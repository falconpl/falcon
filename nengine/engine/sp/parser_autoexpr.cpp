/*
   FALCON - The Falcon Programming Language.
   FILE: parser_autoexpr.cpp

   Parser for Falcon source files -- autoexpression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_autoexpr.cpp"

#include <falcon/setup.h>

#include <falcon/expression.h>
#include <falcon/exprsym.h>
#include <falcon/statement.h>
#include <falcon/error.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_autoexpr.h>

#include "private_types.h"


namespace Falcon {

using namespace Parsing;

void apply_line_expr( const Rule&, Parser& p )
{
   TokenInstance* ti = p.getNextToken();
   Expression* expr = static_cast<Expression*>(ti->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* line = new StmtAutoexpr(expr, ti->line(), ti->chr());
   st->addStatement( line );

   // clear the stack
   p.simplify(2);
}

void apply_autoexpr_list( const Rule&, Parser& p )
{
   TokenInstance* ti = p.getNextToken();
   Expression* expr = static_cast<Expression*>(ti->detachValue());
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   ctx->addStatement( new StmtAutoexpr( expr, ti->line(), ti->chr() ) );

   // clear the stack
   p.simplify(2);
}


void apply_stmt_assign_list( const Rule&, Parser& p )
{
   // << (r_Expr_assign << "Expr_assign" << apply_expr_assign << NeListExpr << T_EqSign << NeListExpr)
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   // get the tokens
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();
   //TokenInstance* v1 = p.getNextToken();
   //p.getNextToken();
   TokenInstance* v3 = p.getNextToken();

   // do not detach, we're discarding the list.
   List* listRight = static_cast<List*>(v2->asData());
   List* listLeft = static_cast<List*>(v3->asData());

   fassert( ! listLeft->empty() );
   fassert( ! listRight->empty() );

   TokenInstance *ti = new TokenInstance( v3->line(), v3->chr(), sp.S_MultiAssign );

   // simplify the code down by considering the first token an element of the list
   //listRight->push_back(static_cast<Expression*>( v1->detachValue() ) );
   // do we have just one assignee?
   if( listLeft->size() == 1 )
   {
      if( listRight->size() == 1 )
      {
         // a simple assignment
         ExprAssign* assign = new ExprAssign( listRight->front(), listLeft->front() );
         listRight->clear();
         ti->setValue( assign, expr_deletor );
      }
      else
      {
         ExprUnpack* unpack = new ExprUnpack( listLeft->front(), true );
         // save the unpack already. Even on error, it WAS a try to unpack.
         ti->setValue( unpack, expr_deletor );

         // we abandoned the data in the list
         listLeft->clear();
         List::iterator iterRight = listRight->begin();
         while( iterRight != listRight->end() )
         {
            Expression* expr = *iterRight;
            if( expr->type() != Expression::t_symbol )
            {
               p.addError(e_syn_unpack, p.currentSource(), v2->line(), v2->chr());
               p.simplify(3, ti);
               return;
            }

            // accept this item -- abandon it from the list
            ctx->defineSymbols(expr);
            unpack->addAssignand(static_cast<ExprSymbol*>(expr)->symbol());
            //++iterRight;
         }
          // don't clear the right side list, we got the symbols -- let the expr to die
      }
   }
   else
   {
      // save the unpack already. Even on error, it WAS a try to unpack.
      ExprMultiUnpack* unpack = new ExprMultiUnpack( true );
      ti->setValue( unpack, expr_deletor );

      // multiple assignment
      if( listRight->size() != listLeft->size() )
      {
         // Use second token position to signal the error
         // notice that ti value is now in listRight, so it will be destroyed
         p.addError( e_unpack_size, p.currentSource(), v2->line(), v2->chr() );
         delete ti;
         p.simplify(3);
         return;
      }

      List::iterator iterRight = listRight->begin();
      while( iterRight != listRight->end() )
      {
         Expression* expr = *iterRight;
         if( expr->type() != Expression::t_symbol )
         {
            p.addError(e_syn_unpack, p.currentSource(), v2->line(), v2->chr());
            p.simplify(3, ti);
            return;
         }

         fassert( ! listLeft->empty() );
         Expression* assignand = listLeft->front();
         listLeft->pop_front();

         ctx->defineSymbols(expr);
         unpack->addAssignment(
            static_cast<ExprSymbol*>(expr)->symbol(), assignand );
         ++iterRight;

      }
      fassert( listLeft->empty() );

      // let the simplify to kill the symbol expressions
   }

   p.simplify(3, ti); // actually it has no value
}

}

/* end of parser_autoexpr.cpp */
