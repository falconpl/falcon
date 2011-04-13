/*
   FALCON - The Falcon Programming Language.
   FILE: sourceparser.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/sourceparser.h>
#include <falcon/statement.h>
#include <falcon/parser/nonterminal.h>
#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sourcetokens.h>
#include <falcon/syntree.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/parser/state.h>

#include <falcon/parser/tint.h>
#include <falcon/parser/teol.h>
#include <falcon/parser/teof.h>

namespace Falcon {
using namespace Parsing;

static void expr_deletor(void* data)
{
   Expression* expr = static_cast<Expression*>(data);
   delete expr;
}

   NonTerminal expr0( "Expr0" );
   NonTerminal expr( "Expr" );


//==========================================================
// NonTerminal - Expr0
//==========================================================


//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr0_times( const Rule& r, Parser& p )
{
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), expr0);
   ti->setValue( new ExprTimes(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr0_div( const Rule& r, Parser& p )
{
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(),expr0);
   ti->setValue( new ExprDiv(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v1->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr0_neg( const Rule& r, Parser& p )
{
   TokenInstance* minus = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), expr0);
   ti2->setValue( new ExprNeg(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

static void apply_expr0_int( const Rule& r, Parser& p )
{
   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), expr0);
   ti2->setValue( new ExprValue(ti->asInteger()), expr_deletor );
   p.simplify(1,ti2);
}


//==========================================================
// NonTerminal - Expr1
//==========================================================


//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr_plus( const Rule& r, Parser& p )
{
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), expr);
   ti->setValue( new ExprPlus(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_minus( const Rule& r, Parser& p )
{
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), expr);
   ti->setValue( new ExprMinus(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_pars( const Rule& r, Parser& p )
{
   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), expr);
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(3,ti2);
}

static void apply_expr_expr0( const Rule& r, Parser& p )
{
   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(),expr);
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(1,ti2);
}

//==========================================================
// A line
//==========================================================


static void apply_line_expr( const Rule& r, Parser& p )
{
   TokenInstance* ti = p.getNextToken();
   Expression* expr = static_cast<Expression*>(ti->detachValue());
   SynTree* st = static_cast<SynTree*>(p.context());

   Statement* line = new StmtAutoexpr(expr, ti->line(), ti->chr());
   st->append( line );
   
   // clear the stack
   p.simplify(2);
}

//==========================================================
// States
//==========================================================

//==========================================================
// SourceParser
//==========================================================


SourceParser::SourceParser( SynTree* st ):
   m_syntree(st)
{
   m_ctx = st;
}

bool SourceParser::parse()
{
   Rule r_expr0_0 = Rule::Maker( "Expr0_times", apply_expr0_times ).t(expr0).t(t_times).t(expr0);
   Rule r_expr0_1 = Rule::Maker( "Expr0_div", apply_expr0_div ).t(expr0).t(t_divide).t(expr0);
   Rule r_expr0_2 = Rule::Maker( "Expr0_neg", apply_expr0_neg ).t(t_minus).t(expr0);
   Rule r_expr0_3 = Rule::Maker( "Expr0_number", apply_expr0_int ).t(t_int);

   Rule r_expr_0 = Rule::Maker( "Expr_plus", apply_expr_plus ).t(expr).t(t_plus).t(expr);
   Rule r_expr_1 = Rule::Maker( "Expr_minus", apply_expr_minus ).t(expr).t(t_minus).t(expr);
   Rule r_expr_2 = Rule::Maker( "Expr_pars", apply_expr_pars ).t(t_openpar).t(expr).t(t_closepar);
   Rule r_expr_3 = Rule::Maker( "Expr_from_expr0", apply_expr_expr0 ).t(expr0);

   Rule r_line_expr = Rule::Maker( "line_expr", apply_line_expr ).t(expr).t(t_eol);
   NonTerminal nt_line = NonTerminal::Maker( "LINE" ).r(r_line_expr);

   Rule r_line_eof = Rule::Maker( "line_expr", apply_line_expr ).t(expr).t(t_eof);
   NonTerminal nt_eof = NonTerminal::Maker( "LINE_EOF" ).r(r_line_eof);

   // prepare expr0
   expr0.r(r_expr0_0).r(r_expr0_1).r(r_expr0_2).r(r_expr0_3);

   // prepare expr1
   expr.r(r_expr_0).r(r_expr_1).r(r_expr_2).r(r_expr_3);

   State sMain = State::Maker("main").n(nt_line).n(nt_eof);

   addState(sMain);
   
   return Parser::parse("main");
}

}

/* end of sourceparser.cpp */
