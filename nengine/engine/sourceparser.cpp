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
#include <falcon/parser/state.h>

#include <falcon/syntree.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>

#include <falcon/globalsymbol.h>
#include <falcon/localsymbol.h>

#include "falcon/parsercontext.h"

#include <deque>

namespace Falcon {
using namespace Parsing;

typedef std::deque<Expression*> List;

static void expr_deletor(void* data)
{
   Expression* expr = static_cast<Expression*>(data);
   delete expr;
}

static void list_deletor(void* data)
{
   List* expr = static_cast<List*>(data);
   List::iterator iter = expr->begin();
   while( iter != expr->end() )
   {
      delete *iter;
      ++iter;
   }
   delete expr;
}

//==========================================================
// NonTerminal - Expr
//==========================================================

//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr_assign( const Rule& r, Parser& p )
{
   // << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   Expression* firstPart = static_cast<Expression*>(v1->detachValue());
   ctx->defineSymbols(firstPart);
   
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprAssign(
         firstPart,
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}


static void apply_expr_equal( const Rule& r, Parser& p )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprEQ(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_diff( const Rule& r, Parser& p )
{
   // << (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprNE(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}


static void apply_expr_eeq( const Rule& r, Parser& p )
{
   // << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprEEQ(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}


static void apply_expr_call( const Rule& r, Parser& p )
{
   // r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   // TODO: read the expressions in the pars
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ExprCall* call = new ExprCall( static_cast<Expression*>(v1->detachValue()) );

   List* list = static_cast<List*>(v2->detachValue());
   List::iterator iter = list->begin();
   while( iter != list->end() )
   {
      call->addParameter( *iter );
      ++iter;
   }
   // free the expressions in the list
   list->clear();

   ti->setValue( call, expr_deletor );
   p.simplify(4,ti);
}


static void apply_expr_index( const Rule& r, Parser& p )
{
   // << (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprIndex(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(4,ti);
}

static void apply_expr_star_index( const Rule& r, Parser& p )
{
   // << (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();
   p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprStarIndex(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );

   p.simplify(5,ti);
}



static void apply_expr_dot( const Rule& r, Parser& p )
{
   // << (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprDot(
         *v2->asString(),
         static_cast<Expression*>(v1->detachValue())
      ), expr_deletor );

   p.simplify(3,ti);
}



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



//typedef void(*Apply)( const Rule& r, Parser& p );
static void apply_expr_times( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprTimes(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr_div( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr );
   ti->setValue( new ExprDiv(
         static_cast<Expression*>(v1->detachValue()),
         static_cast<Expression*>(v2->detachValue())
      ), expr_deletor );
   p.simplify(3,ti);
}

static void apply_expr_neg( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* minus = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprNeg(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

static void apply_expr_atom( const Rule& r, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Expr );
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(1,ti2);
}

//==========================================================
// NonTerminal - Atom
//==========================================================

static void apply_Atom_Int ( const Rule& r, Parser& p )
{
   // << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )

   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(ti->asInteger()), expr_deletor );
   p.simplify(1,ti2);
}

static void apply_Atom_Float ( const Rule& r, Parser& p )
{
   // << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(ti->asNumeric()), expr_deletor );
   p.simplify(1,ti2);
}

static void apply_Atom_Name ( const Rule& r, Parser& p )
{
   // << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* ti = p.getNextToken();
   Symbol* sym = ctx->addVariable(*ti->asString());

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( sym->makeExpression(), expr_deletor );
   p.simplify(1,ti2);
}

static void apply_Atom_String ( const Rule& r, Parser& p )
{
   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Atom );

   // get the string and it's class, to generate a static UserValue
   String* s = ti->detachString();
   Class* stringClass = Engine::instance()->stringClass();
   // tell the context that we have a new string around.
   ctx->onStaticData( stringClass, s );

   // set it in the expression
   Item itm;
   itm.setUser( stringClass, s );
   ti2->setValue( new ExprValue(itm), expr_deletor );

   // remove the token in the stack.
   p.simplify(1,ti2);
}

static void apply_Atom_Nil ( const Rule& r, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(Item()), expr_deletor );
   p.simplify(1,ti2);
}

//==========================================================
// Statements
//==========================================================


static void apply_line_expr( const Rule& r, Parser& p )
{
   TokenInstance* ti = p.getNextToken();
   Expression* expr = static_cast<Expression*>(ti->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* line = new StmtAutoexpr(expr, ti->line(), ti->chr());
   st->addStatement( line );

   // clear the stack
   p.simplify(2);
}


static void apply_if_short( const Rule& r, Parser& p )
{
   // << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon << S_Autoexpr << T_EOL )

   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();
   TokenInstance* tstatement = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   Expression* sa = static_cast<Expression*>(tstatement->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* ifTrue = new SynTree;
   ifTrue->append( new StmtAutoexpr(sa) );

   StmtIf* stmt_if = new StmtIf(expr, ifTrue, 0, tif->line(), tif->chr());
   st->addStatement( stmt_if );

   // clear the stack
   p.simplify(5);
}

static void apply_if( const Rule& r, Parser& p )
{
   // << (r_if << "if" << apply_if << T_if << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* ifTrue = new SynTree;
   StmtIf* stmt_if = new StmtIf(expr, ifTrue, 0, tif->line(), tif->chr());
   st->openBlock( stmt_if, ifTrue );

   // clear the stack
   p.simplify(3);
}

static void apply_elif( const Rule& r, Parser& p )
{
   // << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   if( current == 0 || current->type() != Statement::if_t )
   {
      p.addError( e_syn_elif, p.currentSource(), tif->line(), tif->chr() );
      delete expr;
   }
   else
   {
      StmtIf* stmt_if = static_cast<StmtIf*>(current);
      SynTree* ifElse = new SynTree;
      stmt_if->addElif( expr, ifElse, tif->line(), tif->chr() );
      st->changeBranch( ifElse );
   }

   // clear the stack
   p.simplify(3);
}

static void apply_else( const Rule& r, Parser& p )
{
   // << (r_else << "else" << apply_else << T_else << T_EOL )
   TokenInstance* telse = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   if( current == 0 || current->type() != Statement::if_t )
   {
      p.addError( e_syn_else, p.currentSource(), telse->line(), telse->chr() );
   }
   else
   {
      StmtIf* stmt_if = static_cast<StmtIf*>(current);
      SynTree* ifElse = new SynTree;
      stmt_if->setElse( ifElse );
      st->changeBranch( ifElse );
   }

   // clear the stack
   p.simplify(2);
}


static void apply_while_short( const Rule& r, Parser& p )
{
   // << (r_while_short << "while_short" << apply_while_short << T_while << Expr << T_Colon << Expr << T_EOL )

   TokenInstance* twhile = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();
   TokenInstance* tstatement = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   Expression* sa = static_cast<Expression*>(tstatement->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* whsyn = new SynTree;
   whsyn->append( new StmtAutoexpr(sa) );

   StmtWhile* stmt_wh = new StmtWhile(expr, whsyn);
   stmt_wh->decl( twhile->line(), twhile->chr() );
   st->addStatement( stmt_wh );

   // clear the stack
   p.simplify(5);
}

static void apply_while( const Rule& r, Parser& p )
{
   // << (r_while << "while" << apply_while << T_while << Expr << T_EOL )
   TokenInstance* twhile = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* whsyn = new SynTree;
   StmtWhile* stmt_while = new StmtWhile(expr, whsyn );
   stmt_while->decl( twhile->line(), twhile->chr() );
   st->openBlock( stmt_while, whsyn );

   // clear the stack
   p.simplify(3);
}

static void apply_end( const Rule& r, Parser& p )
{
   // << (r_end << "end" << apply_end << T_end << T_EOL )
   TokenInstance* tend = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   if( current == 0 )
   {
      p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
   }
   else
   {
      st->closeContext();
   }

   // clear the stack
   p.simplify(2);
}

static void apply_end_rich( const Rule& r, Parser& p )
{
   // << (r_end_rich << "RichEnd" << apply_end_rich << T_end << Expr << T_EOL )
   TokenInstance* tend = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   // TODO: Actually, it's used for the Loop statement.
   if( current == 0 )
   {
      p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
   }
   else
   {
      delete expr; // todo; actually put in loop
      st->closeContext();
   }

   // clear the stack
   p.simplify(3);
}

//==========================================================
// Lists
//==========================================================

static void apply_ListExpr_next( const Rule& r, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   List* list = static_cast<List*>(tlist->detachValue());
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ListExpr );
   ti_list->setValue( list, list_deletor );
   p.simplify(3,ti_list);

}

static void apply_ListExpr_first( const Rule& r, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();
   
   Expression* expr = static_cast<Expression*>(texpr->detachValue());;

   List* list = new List;
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(texpr->line(), texpr->chr(), sp.ListExpr );
   ti_list->setValue( list, list_deletor );

   // Change the expression in a list
   p.simplify( 1, ti_list );
}

static void apply_ListExpr_empty( const Rule& r, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = new TokenInstance(0, 0, sp.ListExpr );

   List* list = new List;
   ti_list->setValue( list, list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}

//==========================================================
// SourceParser
//==========================================================



SourceParser::SourceParser():
   T_Openpar("(",10),
   T_Closepar(")"),
   T_OpenSquare("[",10),
   T_CloseSquare("]"),
   T_OpenGraph("{",10),
   T_CloseGraph("}"),

   T_Dot(".", 20, true),
   T_Comma( "," , 180 ),

   T_Power("**", 25),

   T_Times("*",30),
   T_Divide("/",30),
   T_Modulo("%",30),

   T_Plus("+",50),
   T_Minus("-",50),

   T_DblEq("==", 70),
   T_NotEq("!=", 70),
   T_Colon( ":" ),
   T_EqSign("=", 200, true),


   T_as("as"),
   T_eq("eq", 70 ),
   T_if("if"),
   T_in("in", 20),
   T_or("or", 130),
   T_to("to", 70),
   
   T_and("and", 120),
   T_def("def"),
   T_end("end"),
   T_for("for"),
   T_not("not", 50),
   T_nil("nil"),
   T_try("try"),

   T_elif("elif"),
   T_else("else"),

   T_while("while")
{
   S_Autoexpr << "Autoexpr"
      << (r_line_autoexpr << "Autoexpr" << apply_line_expr << Expr << T_EOL)
      ;

   S_If << "IF"
      << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon << Expr << T_EOL )
      << (r_if << "if" << apply_if << T_if << Expr << T_EOL )
      ;

   S_Elif << "ELIF"
      << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
      ;

   S_Else << "ELSE"
      << (r_else << "else" << apply_else << T_else << T_EOL )
      ;

   S_While << "WHILE"
      << (r_while_short << "while_short" << apply_while_short << T_while << Expr << T_Colon << Expr << T_EOL )
      << (r_while << "while" << apply_while << T_while << Expr << T_EOL )
      ;

   S_End << "END"
      << (r_end << "end" << apply_end << T_end << T_EOL )
      << (r_end_rich << "RichEnd" << apply_end_rich << T_end << Expr << T_EOL )
      ;

   //==========================================================================
   // Expression
   //
   Expr << "Expr"
      << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << Expr)

      << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
      << (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr)
      << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)

      << (r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar )
      << (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare )
      << (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare )
      << (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name)
      << (r_Expr_plus << "Expr_plus" << apply_expr_plus << Expr << T_Plus << Expr)
      << (r_Expr_minus << "Expr_minus" << apply_expr_minus << Expr << T_Minus << Expr)
      << (r_Expr_pars << "Expr_pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar)      
      << (r_Expr_times << "Expr_times" << apply_expr_times << Expr << T_Times << Expr)
      << (r_Expr_div   << "Expr_div"   << apply_expr_div   << Expr << T_Divide << Expr )
      << (r_Expr_neg   << "Expr_neg"   << apply_expr_neg << T_Minus << Expr )
      << (r_Expr_Atom << "Expr_atom" << apply_expr_atom << Atom)
      ;

   Atom << "Atom"
      << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )
      << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
      << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
      << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
      << (r_Atom_Nil<< "Atom_Nil" << apply_Atom_Nil << T_nil )
      ;

   ListExpr << "ListExpr"
      << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
      << (r_ListExpr_first << "ListExpr_first" << apply_ListExpr_first << Expr )
      << (r_ListExpr_empty << "ListExpr_empty" << apply_ListExpr_empty )
      ;


   //==========================================================================
   //State declarations
   //
   s_Main << "Main"
      << S_Autoexpr
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_End
      ;

   addState( s_Main );
   
}

bool SourceParser::parse()
{
   // we need a context (and better to be a SourceContext
   if ( m_ctx == 0 )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, __FILE__ ).extra("SourceParser::parse - setContext") );
   }

   return Parser::parse("Main");
}



}

/* end of sourceparser.cpp */
