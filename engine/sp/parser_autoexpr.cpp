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
#include <falcon/error.h>
#include <falcon/symbol.h>

#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_autoexpr.h>

#include <falcon/psteps/exprassign.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprunpack.h>
#include <falcon/psteps/exprmultiunpack.h>
#include <falcon/synclasses_id.h>

#include "private_types.h"


namespace Falcon {

using namespace Parsing;

// About the error handler for the autoexpr statement,
// it is regarded as a standard syntax error, so it's left to the main
// parser error detectio routine to be cleared.

static bool check_type( Parser&p, Expression* expr, int line, int chr )
{
   // assignable expressions are only:
   // -- symbols
   // -- accessors
   // -- calls (i.e. if they return a reference).
   Expression::t_trait type = expr->trait();
   if( type != Expression::e_trait_symbol &&
       type != Expression::e_trait_assignable
     )
   {
     p.addError( e_assign_sym, p.currentSource(), line, chr, 0 );
     return false;
   }
   return true;
}


void apply_line_expr( const NonTerminal&, Parser& p )
{
   TokenInstance* ti = p.getNextToken();

   // in interactive context, add the statement only if we don't have errors.
   if( !p.interactive() || p.lastErrorLine() < ti->line() )
   {
      ParserContext* ctx = static_cast<ParserContext*>(p.context());      
      Expression* expr = static_cast<Expression*>(ti->detachValue());
      ctx->accessSymbols( expr );
      ctx->addStatement( expr );
   }

   // clear the stack
   p.simplify(2);
}

void apply_autoexpr_list( const NonTerminal&r, Parser& p )
{
   apply_line_expr( r, p );
}


void apply_stmt_assign_list( const NonTerminal&, Parser& p )
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
   List* listLeft = static_cast<List*>(v2->asData());
   List* listRight = static_cast<List*>(v3->asData());

   fassert( ! listRight->empty() );
   fassert( ! listLeft->empty() );

   TokenInstance *ti = TokenInstance::alloc( v3->line(), v3->chr(), sp.S_MultiAssign );

   // simplify the code down by considering the first token an element of the list
   //listRight->push_back(static_cast<Expression*>( v1->detachValue() ) );
   // do we have just one assignee?
   if( listRight->size() == 1 )
   {
      if( listLeft->size() == 1 )
      {
         // a simple assignment
         check_type(p, listLeft->front(), v2->line(), v2->chr() );
         // create the expression even on failure.
         Expression* first = listLeft->front();
         Expression* second = listRight->front();
         // first access...
         ctx->accessSymbols(second);
         // then define
         ctx->defineSymbols(first);

         ExprAssign* assign = new ExprAssign( first, second, v2->line(), v2->chr() );
         listLeft->clear();
         ti->setValue( assign, treestep_deletor );
      }
      else
      {
         ctx->accessSymbols(listRight->front());
         ExprUnpack* unpack = new ExprUnpack( listRight->front(), v2->line(), v2->chr() );
         // save the unpack already. Even on error, it WAS a try to unpack.
         ti->setValue( unpack, treestep_deletor );

         // we abandoned the data in the list
         listRight->clear();
         List::iterator iterRight = listLeft->begin();
         while( iterRight != listLeft->end() )
         {
            Expression* expr = *iterRight;

            // accept this item -- abandon it from the list
            ctx->defineSymbols(expr);
            if (! unpack->append(expr) )
            {
               sp.addError(e_assign_sym, p.currentSource(), expr->line(), expr->chr());
               delete expr; // the expression was not accepted
            }
            ++iterRight;
         }
         // we need the expressions that we kept...
         listLeft->clear();
      }
   }
   else
   {
      // save the unpack already. Even on error, it WAS a try to unpack.
      ExprMultiUnpack* unpack = new ExprMultiUnpack( v2->line(), v2->chr() );
      ti->setValue( unpack, treestep_deletor );

      // multiple assignment
      if( listLeft->size() != listRight->size() )
      {
         // Use second token position to signal the error
         p.addError( e_unpack_size, p.currentSource(), v3->line(), v3->chr() );
         p.simplify(3, ti);
         return;
      }

      List::iterator iterRight = listLeft->begin();
      while( iterRight != listLeft->end() )
      {
         Expression* expr = *iterRight;
         if ( ! check_type( p, expr, v2->line(), v2->chr() ) )
         {
            p.simplify(3, ti);
            return;
         }         

         fassert( ! listRight->empty() );
         Expression* assignand = listRight->front();
         ctx->accessSymbols(assignand);
         listRight->pop_front();

         ctx->defineSymbols(expr);
         if( expr->lvalueStep() == 0 )
         {
            sp.addError(e_assign_sym, p.currentSource(), expr->line(), expr->chr());
            // but keep the expression, as ExprAssing will accept it just the same.
         }
         unpack->append( new ExprAssign( expr, assignand, v2->line(), v2->chr() ) );
         ++iterRight;

      }
      fassert( listRight->empty() );

      // don't let the simplify to kill the expressions
      listLeft->clear();
   }

   p.simplify(3, ti); // actually it has no value
}

}

/* end of parser_autoexpr.cpp */
