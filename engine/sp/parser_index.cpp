/*
   FALCON - The Falcon Programming Language.
   FILE: parser_index.cpp

   Parser for Falcon source files -- index accessor handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:04:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_index.cpp"

#include <falcon/setup.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_index.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprindex.h>
#include <falcon/psteps/exprstarindex.h>
#include <falcon/psteps/exprrange.h>

namespace Falcon {

using namespace Parsing;


void apply_expr_index( const NonTerminal&, Parser& p )
{
   // << (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprIndex( static_cast<Expression*>( v1->detachValue() ),
         static_cast<Expression*>( v2->detachValue() ),
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 4, ti );
}


void apply_expr_star_index( const NonTerminal&, Parser& p )
{
   // << (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprStarIndex( static_cast<Expression*>( v1->detachValue() ),
         static_cast<Expression*>( v2->detachValue() ),
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 5, ti );
}


void apply_expr_range_index3( const NonTerminal&, Parser& p )
{
   // << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare
   // Example array[n1:n2:n3]
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();  // array
   p.getNextToken(); // '['
   TokenInstance* tstart = p.getNextToken(); // n1
   p.getNextToken(); // ':'
   TokenInstance* tend = p.getNextToken();   // n2
   p.getNextToken(); // ':'
   TokenInstance* tstep = p.getNextToken();  // n3
   
   ExprRange* rng = new ExprRange( static_cast<Expression*>( tstart->detachValue() ),
         static_cast<Expression*>( tend->detachValue() ),
         static_cast<Expression*>( tstep->detachValue() ),
         v1->line(),
         v1->chr()
      );
   
   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprIndex( static_cast<Expression*>( v1->detachValue() ),
         rng,
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 8, ti );
}


void apply_expr_range_index3open( const NonTerminal&, Parser& p )
{
   // << Expr << T_OpenSquare << Expr << T_Colon << T_Colon << Expr << T_CloseSquare
   // Example: array[n1::n2]
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();  // array
   p.getNextToken(); // '['
   TokenInstance* tstart = p.getNextToken(); // n1
   p.getNextToken(); // ':'
   p.getNextToken(); // ':'
   TokenInstance* tstep = p.getNextToken();  // n2
   
   ExprRange* rng = new ExprRange( 
      static_cast<Expression*>( tstart->detachValue() ),
      0,
      static_cast<Expression*>( tstep->detachValue() ) );
   
   TokenInstance* ti = TokenInstance::alloc(v1->line(), v1->chr(), sp.Expr);

   ti->setValue( new ExprIndex( static_cast<Expression*>( v1->detachValue() ),
         rng,
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 7, ti );   
}


void apply_expr_range_index2( const NonTerminal&, Parser& p )
{
   // << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_CloseSquare
   // Example array[n1:n2]
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();  // array
   p.getNextToken(); // '['
   TokenInstance* tstart = p.getNextToken(); // n1
   p.getNextToken(); // ':'
   TokenInstance* tend = p.getNextToken();   // n2
   
   ExprRange* rng = new ExprRange( static_cast<Expression*>( tstart->detachValue() ),
      static_cast<Expression*>( tend->detachValue() ),
      0,
      v1->line(),
      v1->chr() );
   
   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprIndex( static_cast<Expression*>( v1->detachValue() ),
         rng,
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 6, ti );
}


void apply_expr_range_index1( const NonTerminal&, Parser& p )
{
   // << Expr << T_OpenSquare << Expr << T_Colon << T_CloseSquare
   // Example: array[n1:]
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();  // "array"
   p.getNextToken(); // '['
   TokenInstance* tstart = p.getNextToken(); // "n1"
   
   ExprRange* rng = new ExprRange( static_cast<Expression*>( tstart->detachValue() ),
      0,
      0,
      v1->line(),
      v1->chr() );
   
   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprIndex(
         static_cast<Expression*>( v1->detachValue() ),
            rng,
            v1->line(), v1->chr() ),
      treestep_deletor );

   p.simplify(5,ti);
}


void apply_expr_range_index0( const NonTerminal&, Parser& p )
{
   // << Expr << T_OpenSquare << T_Colon << T_CloseSquare
   // Example: array[:]
   SourceParser& sp = static_cast<SourceParser&>( p );

   TokenInstance* v1 = p.getNextToken();  // array

   ExprRange* rng = new ExprRange( 
         0,
         0,
         0,
         v1->line(), v1->chr()
      );

   TokenInstance* ti = TokenInstance::alloc( v1->line(), v1->chr(), sp.Expr );

   ti->setValue( new ExprIndex( static_cast<Expression*>( v1->detachValue() ),
         rng,
         v1->line(),
         v1->chr() ),
      treestep_deletor );

   p.simplify( 4, ti );
}


}

/* end of parser_index.cpp */
