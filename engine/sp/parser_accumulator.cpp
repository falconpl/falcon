/*
   FALCON - The Falcon Programming Language.
   FILE: parser_accumulator.cpp

   Parser for Falcon source files -- Accumulator expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 04 Apr 2013 22:28:34 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_function.cpp"

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/error.h>

#include <falcon/parser/parser.h>
#include <falcon/psteps/exprep.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_accumulator.h>

#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/expraccumulator.h>

#include "private_types.h"
#include <falcon/psteps/exprsym.h>
#include <falcon/classes/classsyntree.h>
#include <falcon/psteps/exprep.h>
#include <falcon/stdhandlers.h>


namespace Falcon {

using namespace Parsing;

void apply_expr_accumulator( const NonTerminal&, Parser& p)
{
   // T_CapSquare << AccumulatorBody

   SourceParser& sp = *static_cast<SourceParser*>(&p);

   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   //ExprAccumulator* ea = new ExprAccumulator(ti->line(),ti->chr());
   ti->token( sp.Expr );

   p.simplify(1); // keep accumulator
}


static void internal_accumulator( int count, int line, int c, Parser& p, List* l,  Expression* filter, Expression* target )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ExprAccumulator* accumulator = new ExprAccumulator( line, c );

   size_t len = l->size();
   if( len != 0 )
   {
      ExprEP* ep = new ExprEP(line,c);
      for( size_t pos = 0; pos < len; ++pos )
      {
         Expression* expr = l->at(pos);
         ep->append(expr);
      }

      accumulator->selector(ep);
   }

   if( filter != 0 )
   {
      accumulator->filter(filter);
   }

   if( target != 0 )
   {
      accumulator->target(target);
   }

   TokenInstance* tinst = TokenInstance::alloc( line, c, sp.AccumulatorBody );
   tinst->setValue( accumulator, &treestep_deletor );

   p.simplify(count, tinst); // keep accumulator
}


void apply_accumulator_complete( const NonTerminal&, Parser& p)
{
   //<< ListExpr << T_CloseSquare << Expr <<  T_Arrow << Expr
   TokenInstance* ti_listexpr = p.getNextToken();
   p.getNextToken(); // T_CloseSquare
   TokenInstance* ti_filter = p.getNextToken();
   p.getNextToken(); // T_Arrow
   TokenInstance* ti_target = p.getNextToken();

   internal_accumulator( 5, ti_listexpr->line(), ti_listexpr->chr(), p,
            static_cast<List*>(ti_listexpr->detachValue()),
            static_cast<Expression*>(ti_filter->detachValue()),
            static_cast<Expression*>(ti_target->detachValue())
            );
}


void apply_accumulator_w_target( const NonTerminal&, Parser& p)
{
   //<< ListExpr << T_CloseSquare << T_Arrow << Expr
   TokenInstance* ti_listexpr = p.getNextToken();
   p.getNextToken(); // T_CloseSquare
   p.getNextToken(); // T_Arrow
   TokenInstance* ti_target = p.getNextToken();

   internal_accumulator( 4, ti_listexpr->line(), ti_listexpr->chr(), p,
            static_cast<List*>(ti_listexpr->detachValue()),
            0,
            static_cast<Expression*>(ti_target->detachValue() )
            );
}


void apply_accumulator_w_filter( const NonTerminal&, Parser& p)
{
   //<< ListExpr << T_CloseSquare << Expr
   TokenInstance* ti_listexpr = p.getNextToken();
   p.getNextToken(); // T_CloseSquare
   TokenInstance* ti_filter = p.getNextToken();

   internal_accumulator( 3, ti_listexpr->line(), ti_listexpr->chr(), p,
            static_cast<List*>(ti_listexpr->detachValue()),
            static_cast<Expression*>(ti_filter->detachValue() ),
            0
            );
}


void apply_accumulator_simple( const NonTerminal&, Parser& p )
{
   //<< ListExpr << T_CloseSquare
   TokenInstance* ti_listexpr = p.getNextToken();

   internal_accumulator( 2, ti_listexpr->line(), ti_listexpr->chr(), p,
            static_cast<List*>(ti_listexpr->detachValue()),
            0, 0 );
}

}

/* end of parser_accumulator.cpp */
