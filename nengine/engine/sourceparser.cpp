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

#include <falcon/syntree.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/parser/state.h>

namespace Falcon {
using namespace Parsing;

static void expr_deletor(void* data)
{
   Expression* expr = static_cast<Expression*>(data);
   delete expr;
}

//==========================================================
// NonTerminal - Expr0
//==========================================================


//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr0_times( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr0);
   ti->setValue( new ExprTimes(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr0_div( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr0 );
   ti->setValue( new ExprDiv(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr0_neg( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* minus = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), sp.Expr0 );
   ti2->setValue( new ExprNeg(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

static void apply_expr0_int( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Expr0 );
   ti2->setValue( new ExprValue(ti->asInteger()), expr_deletor );
   p.simplify(1,ti2);
}


//==========================================================
// NonTerminal - Expr1
//==========================================================


//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr_plus( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprPlus(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_minus( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprMinus(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_pars( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Expr);
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(3,ti2);
}

static void apply_expr_expr0( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Expr);
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
// SourceParser
//==========================================================


SourceParser::SourceParser( SynTree* st ):
   T_Plus("+"),
   T_Times("*"),
   T_Divide("/"),
   T_Minus("-"),
   T_Modulo("%"),
   T_Power("**"),

   T_Openpar("("),
   T_Closepar(")"),
   T_OpenSquare("["),
   T_CloseSquare("]"),
   T_OpenGraph("{"),
   T_CloseGraph("}"),

   T_Dot("."),

   T_as("as"),
   T_and("and"),
   T_end("end"),
   T_eq("eq"),
   T_if("if"),
   T_in("in"),
   T_not("not"),
   T_nil("nil"),
   T_or("or"),
   T_to("to"),
   
   m_syntree(st)
{
   m_ctx = st;

   // Style 1 --- all in one
   Expr0 << "Expr0"
      << (r_Expr0_times << "Expr0_times" << apply_expr0_times << Expr0 << T_Times << Expr0)
      << (r_Expr0_div   << "Expr0_div"   << apply_expr0_div   << Expr0 << T_Divide << Expr0 )
      << (r_Expr0_neg   << "Expr0_neg"   << apply_expr0_neg << T_Minus << Expr0 )
      << (r_Expr0_number<< "Expr0_number"<< apply_expr0_int << T_Int);

   // Style 2 -- bottom-top
   r_Expr_plus << "Expr_plus" << apply_expr_plus << Expr << T_Plus << Expr;
   r_Expr_minus << "Expr_minus" << apply_expr_minus << Expr << T_Minus << Expr;
   r_Expr_pars << "Expr_pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar;
   r_Expr_from_expr0 << "Expr_from_expr0" << apply_expr_expr0 << Expr0;

   Expr << "Expr" << r_Expr_plus << r_Expr_minus <<  r_Expr_pars << r_Expr_from_expr0;

   // Style 3 -- top-down
   Line << "Line" << r_line_autoexpr;
   r_line_autoexpr << "Line_Autoexpr" << apply_line_expr << Expr << T_EOL;


   //State declarations
   s_Main << "Main"
            << Line;

   addState( s_Main );
}

bool SourceParser::parse()
{   
   return Parser::parse("Main");
}



}

/* end of sourceparser.cpp */
