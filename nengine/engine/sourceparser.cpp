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
#include <falcon/exprarray.h>
#include <falcon/exprdict.h>

#include <falcon/globalsymbol.h>
#include <falcon/localsymbol.h>
#include <falcon/parsercontext.h>
#include <falcon/stmtrule.h>
#include <falcon/error.h>
#include <falcon/codeerror.h>
#include <falcon/pseudofunc.h>

#include <deque>

#include <stdio.h>

namespace Falcon {
using namespace Parsing;

typedef std::deque<Expression*> List;
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

static void expr_deletor(void* data)
{
   Expression* expr = static_cast<Expression*>(data);
   delete expr;
}


class PairList: public std::deque< std::pair<Expression*,Expression*> >
{
public:
   PairList():
      m_bHasPairs(false)
   {}

   bool m_bHasPairs;
};

static void pair_list_deletor(void* data)
{
   PairList* expr = static_cast<PairList*>(data);
   PairList::iterator iter = expr->begin();
   while( iter != expr->end() )
   {
      delete iter->first;
      delete iter->second;
      ++iter;
   }
   delete expr;
}

typedef std::deque<String> NameList;

static void name_list_deletor(void* data)
{
   delete static_cast<NameList*>(data);
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

static void apply_expr_call( const Rule& r, Parser& p )
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
   if( callee->type() == Expression::t_symbol )
   {
      // check if the symbol is a pseudofunction
      Symbol* funsym = static_cast<ExprSymbol*>(callee)->symbol();
      PseudoFunction* pf = einst->getPseudoFunction( funsym->name() );

      // if it is, we don't need the callee expression anymore.
      if( pf != 0 )
      {
         call = new ExprCall(pf);
         delete callee;
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
      call->addParam( *iter );
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


static void apply_expr_array_decl( const Rule& r, Parser& p )
{
   // << (r_Expr_index << "Expr_array_decl" << apply_expr_array_decl << T_OpenSquare << ListExprOrPairs << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tSquare = p.getNextToken();
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti = new TokenInstance(tSquare->line(), tSquare->chr(), sp.Expr);

   PairList* list = static_cast<PairList*>(v1->detachValue());
   if( list->m_bHasPairs )
   {
      ExprDict* dict = new ExprDict;
      PairList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         dict->add(iter->first, iter->second);
         ++iter;
      }
      ti->setValue( dict, expr_deletor );
   }
   else
   {
      ExprArray* array = new ExprArray;
      // it's a dictionary declaration
      PairList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         array->add(iter->first);
         ++iter;
      }
      ti->setValue( array, expr_deletor );
   }

   // free the expressions in the list
   list->clear();

   p.simplify(3,ti);
}

static void apply_expr_empty_dict( const Rule& r, Parser& p )
{
   // << T_OpenSquare << T_Arrow << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tSquare = p.getNextToken();
   TokenInstance* ti = new TokenInstance(tSquare->line(), tSquare->chr(), sp.Expr);
   ExprDict* dict = new ExprDict;
   ti->setValue( dict, expr_deletor );
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

//=======================================================
// Standard binary expressions
//

static void apply_expr_binary( const Rule& r, Parser& p, BinaryExpression* bexpr )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   // Todo: set lvalues and define symbols in the module
   TokenInstance* ti = new TokenInstance(v1->line(), v1->chr(), sp.Expr);
   bexpr->first(static_cast<Expression*>(v1->detachValue()));
   bexpr->second(static_cast<Expression*>(v2->detachValue()));
   ti->setValue( bexpr, expr_deletor );

   p.simplify(3,ti);
}

static void apply_expr_equal( const Rule& r, Parser& p )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   apply_expr_binary(r, p, new ExprEQ );
}

static void apply_expr_diff( const Rule& r, Parser& p )
{
   // << (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr)
   apply_expr_binary(r, p, new ExprNE );
}

static void apply_expr_less( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLT );
}

static void apply_expr_greater( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGT );
}

static void apply_expr_le( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLE );
}

static void apply_expr_ge( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGE );
}


static void apply_expr_eeq( const Rule& r, Parser& p )
{
  // << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)
  apply_expr_binary(r, p, new ExprEEQ );
}


static void apply_expr_plus( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPlus );
}

static void apply_expr_minus( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprMinus );
}

static void apply_expr_times( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprTimes );
}

static void apply_expr_div( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprDiv );
}


static void apply_expr_pow( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPow );
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
      // can we really change branch now?
      SynTree* ifElse = st->changeBranch();
      if ( ifElse != 0 ) {
         stmt_if->addElif( expr, ifElse, tif->line(), tif->chr() );
      }
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

      // can we really change branch?
      SynTree* ifElse = st->changeBranch();
      if( ifElse != 0 ) {
         stmt_if->setElse( ifElse );
      }
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


static void apply_rule( const Rule& r, Parser& p )
{
   // << (r_rule << "rule" << apply_rule << T_rule << T_EOL )
   TokenInstance* trule = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   StmtRule* stmt_rule = new StmtRule( trule->line(), trule->chr() );
   st->openBlock( stmt_rule, &stmt_rule->currentTree() );

   // clear the stack
   p.simplify(2);
}

static void apply_cut( const Rule& r, Parser& p )
{
   // << (r_cut << "cut" << apply_cut << T_cut << T_EOL )
   TokenInstance* trule = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   if ( st->currentStmt() == 0 || st->currentStmt()->type() != Statement::rule_t )
   {
      p.addError( e_syn_cut, p.currentSource(), trule->line(), trule->chr() );
   }
   else
   {
      StmtCut* stmt_cut = new StmtCut( trule->line(), trule->chr() );
      static_cast<StmtRule*>(st->currentStmt())->addStatement(stmt_cut);
   }

   // clear the stack
   p.simplify(2);
}


static void apply_end( const Rule& r, Parser& p )
{
   // << (r_end << "end" << apply_end << T_end << T_EOL )
   TokenInstance* tend = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());
   //Statement* current = st->currentStmt();
   if( !st->currentStmt() && !st->currentFunc() && !st->currentClass())
   {
      // can we close a state?
      p.popState();

      //p.addError( e_syn_end, p.currentSource(), tend->line(), tend->chr() );
      return;
   }

   // clear the stack
   p.simplify(2);
   st->closeContext();
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
// PairLists
//==========================================================

static void apply_ListExprOrPairs_next_pair( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_next_pair << "ListExprOrPairs_next_pair" << apply_ListExprOrPairs_next_pair
   //  << ListExprOrPairs << T_Comma << Expr << T_Arrow << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr2 = p.getNextToken();

   Expression* expr1 = static_cast<Expression*>(texpr1->detachValue());
   Expression* expr2 = static_cast<Expression*>(texpr2->detachValue());
   PairList* list = static_cast<PairList*>(tlist->detachValue());
   list->push_back(std::make_pair(expr1,expr2));

   // if we didn't have a list of pairs, declare error in dict decl
   if( ! list->m_bHasPairs )
   {
      p.addError(e_syn_arraydecl, p.currentSource(),
            texpr1->line(), texpr1->chr(),
            tlist->line(), "not a pair");
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ListExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(5,ti_list);

}


static void apply_ListExprOrPairs_next( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_next << "ListExprOrPairs_next" << apply_ListExprOrPairs_next << ListExprOrPairs << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   PairList* list = static_cast<PairList*>(tlist->detachValue());

   // if we add a list of pairs, declare error in dict decl
   if( list->m_bHasPairs )
   {
      p.addError(e_syn_dictdecl, p.currentSource(),
            texpr->line(), texpr->chr(),
            tlist->line(), "not a pair of values");
   }
   else
   {
      // detach and asssign only if the list can accept 0 as the second value
      // Otherwise, we'll have problems during dictionary creation.
      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      list->push_back(std::make_pair<Expression*,Expression*>(expr,0));
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ListExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(3,ti_list);

}

static void apply_ListExprOrPairs_first_pair( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_first_pair << "ListExprOrPairs_first_pair" << apply_ListExprOrPairs_first_pair << Expr << T_Arrow << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr2 = p.getNextToken();

   Expression* expr1 = static_cast<Expression*>(texpr1->detachValue());
   Expression* expr2 = static_cast<Expression*>(texpr2->detachValue());

   PairList* list = new PairList;
   list->push_back(std::make_pair(expr1,expr2));
   list->m_bHasPairs = true;

   TokenInstance* ti_list = new TokenInstance(texpr1->line(), texpr1->chr(), sp.ListExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );

   // Change the expression in a list
   p.simplify( 3, ti_list );
}


static void apply_ListExprOrPairs_first( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_first << "ListExprOrPairs_first" << apply_ListExprOrPairs_first << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   PairList* list = new PairList;
   list->push_back(std::make_pair<Expression*,Expression*>(expr,0));

   TokenInstance* ti_list = new TokenInstance(texpr->line(), texpr->chr(), sp.ListExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );

   // Change the expression in a list
   p.simplify( 1, ti_list );
}

static void apply_ListExprOrPairs_empty( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_empty << "ListExprOrPairs_empty" << apply_ListExprOrPairs_empty )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = new TokenInstance(0, 0, sp.ListExprOrPairs );

   PairList* list = new PairList;
   ti_list->setValue( list, pair_list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}


//==========================================================
// SeqPairList
//==========================================================

static void apply_SeqExprOrPairs_next_pair_cm( const Rule& r, Parser& p )
{
   // << SeqExprOrPairs << T_Comma << Expr << T_Arrow << Expr
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr2 = p.getNextToken();

   Expression* expr1 = static_cast<Expression*>(texpr1->detachValue());
   Expression* expr2 = static_cast<Expression*>(texpr2->detachValue());
   PairList* list = static_cast<PairList*>(tlist->detachValue());
   list->push_back(std::make_pair(expr1,expr2));

   // if we didn't have a list of pairs, declare error in dict decl
   if( ! list->m_bHasPairs )
   {
      p.addError(e_syn_arraydecl, p.currentSource(),
            texpr1->line(), texpr1->chr(),
            tlist->line(), "not a pair");
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(5,ti_list);

}

static void apply_SeqExprOrPairs_next_pair( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_next_pair << "ListExprOrPairs_next_pair" << apply_ListExprOrPairs_next_pair
   //  << ListExprOrPairs << Expr << T_Arrow << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr2 = p.getNextToken();

   Expression* expr1 = static_cast<Expression*>(texpr1->detachValue());
   Expression* expr2 = static_cast<Expression*>(texpr2->detachValue());
   PairList* list = static_cast<PairList*>(tlist->detachValue());
   list->push_back(std::make_pair(expr1,expr2));

   // if we didn't have a list of pairs, declare error in dict decl
   if( ! list->m_bHasPairs )
   {
      p.addError(e_syn_arraydecl, p.currentSource(),
            texpr1->line(), texpr1->chr(),
            tlist->line(), "not a pair");
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(4,ti_list);

}


static void apply_SeqExprOrPairs_next( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_next << "ListExprOrPairs_next" << apply_ListExprOrPairs_next << ListExprOrPairs << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   PairList* list = static_cast<PairList*>(tlist->detachValue());

   // if we add a list of pairs, declare error in dict decl
   if( list->m_bHasPairs )
   {
      p.addError(e_syn_dictdecl, p.currentSource(),
            texpr->line(), texpr->chr(),
            tlist->line(), "not a pair of values");
   }
   else
   {
      // detach and asssign only if the list can accept 0 as the second value
      // Otherwise, we'll have problems during dictionary creation.
      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      list->push_back(std::make_pair<Expression*,Expression*>(expr,0));
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(2,ti_list);

}

static void apply_SeqExprOrPairs_next_cm( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_next << "ListExprOrPairs_next" << apply_ListExprOrPairs_next << SeqExprOrPairs << T_COMMA<< Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   PairList* list = static_cast<PairList*>(tlist->detachValue());

   // if we add a list of pairs, declare error in dict decl
   if( list->m_bHasPairs )
   {
      p.addError(e_syn_dictdecl, p.currentSource(),
            texpr->line(), texpr->chr(),
            tlist->line(), "not a pair of values");
   }
   else
   {
      // detach and asssign only if the list can accept 0 as the second value
      // Otherwise, we'll have problems during dictionary creation.
      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      list->push_back(std::make_pair<Expression*,Expression*>(expr,0));
   }

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );
   p.simplify(3,ti_list);

}

// Differs from apply_ListExprOrPairs_first_pair just for sp.SeqExprOrPairs as simplify type
static void apply_SeqExprOrPairs_first_pair( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_first_pair << "ListExprOrPairs_first_pair" << apply_ListExprOrPairs_first_pair << Expr << T_Arrow << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr2 = p.getNextToken();

   Expression* expr1 = static_cast<Expression*>(texpr1->detachValue());
   Expression* expr2 = static_cast<Expression*>(texpr2->detachValue());

   PairList* list = new PairList;
   list->push_back(std::make_pair(expr1,expr2));
   list->m_bHasPairs = true;

   TokenInstance* ti_list = new TokenInstance(texpr1->line(), texpr1->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );

   // Change the expression in a list
   p.simplify( 3, ti_list );
}

// Differs from apply_ListExprOrPairs_first just for sp.SeqExprOrPairs as simplify type
static void apply_SeqExprOrPairs_first( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_first << "ListExprOrPairs_first" << apply_ListExprOrPairs_first << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   PairList* list = new PairList;
   list->push_back(std::make_pair<Expression*,Expression*>(expr,0));

   TokenInstance* ti_list = new TokenInstance(texpr->line(), texpr->chr(), sp.SeqExprOrPairs );
   ti_list->setValue( list, pair_list_deletor );

   // Change the expression in a list
   p.simplify( 1, ti_list );
}

// Differs from apply_ListExprOrPairs_empty just for sp.SeqExprOrPairs as simplify type
static void apply_SeqExprOrPairs_empty( const Rule& r, Parser& p )
{
   // << (r_ListExprOrPairs_empty << "ListExprOrPairs_empty" << apply_ListExprOrPairs_empty )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = new TokenInstance(0, 0, sp.SeqExprOrPairs );

   PairList* list = new PairList;
   ti_list->setValue( list, pair_list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}

static void apply_ListSymbol_first(const Rule& r,Parser& p)
{
   // << (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   TokenInstance* ti_list = new TokenInstance(tname->line(), tname->chr(), sp.ListSymbol);

   NameList* list = new NameList;
   list->push_back(*tname->asString());
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 1, ti_list );
}

static void apply_ListSymbol_next(const Rule& r,Parser& p)
{
   // << (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->detachValue());
   list->push_back(*tname->asString());

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ListSymbol );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );
}

static void apply_ListSymbol_empty(const Rule& r,Parser& p)
{
   // << (r_ListSymbol_empty << "ListSymbol_empty" << apply_ListSymbol_empty )

   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = new TokenInstance(0, 0, sp.ListSymbol);

   NameList* list=new NameList;
   ti_list->setValue( list, name_list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}


static SynFunc* inner_apply_function( const Rule& r, Parser& p, bool bHasExpr )
{
   //<< (r_Expr_function << "Expr_function" << apply_function << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   sp.getNextToken();//T_function
   TokenInstance* tname=sp.getNextToken();
   sp.getNextToken();// '('
   TokenInstance* targs=sp.getNextToken();
   sp.getNextToken();// ')'
   sp.getNextToken();// '\n' or ':'

   TokenInstance* tstatement = 0;
   int tcount = bHasExpr ? 8 : 6;

   if( bHasExpr )
   {
      tstatement = p.getNextToken();
   }

   // Are we already in a function?
   if( ctx->currentFunc() != 0 )
   {
      p.addError( e_toplevel_func,  p.currentSource(), tname->line(), tname->chr() );
      p.simplify(tcount);
      return 0;
   }

   // check if the symbol is free -- defining an unique symbol
   bool alreadyDef;
   GlobalSymbol* symfunc = ctx->onGlobalDefined( *tname->asString(), alreadyDef );
   if( alreadyDef )
   {
      // not free!
      p.addError( e_already_def,  p.currentSource(), tname->line(), tname->chr(), 0,
         String("at line ").N(symfunc->declaredAt()) );
      p.simplify(tcount);
      return 0;
   }

   // Ok, we took the symbol.
   SynFunc* func=new SynFunc(*tname->asString(),0,tname->line());
   NameList* list=static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   if ( bHasExpr )
   {
      Expression* sa = static_cast<Expression*>(tstatement->detachValue());
      func->syntree().append(new StmtReturn(sa));
   }
   else
   {
      ctx->openFunc(func, symfunc);
   }

   p.simplify(tcount);
}

static void apply_function(const Rule& r,Parser& p)
{
   inner_apply_function( r, p, false );
}


static void apply_function_short(const Rule& r,Parser& p)
{
   inner_apply_function( r, p, true );
}

static void apply_return(const Rule& r,Parser& p)
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   sp.getNextToken();//T_function
   TokenInstance* texpr=sp.getNextToken();
   sp.getNextToken();//T_EOL

   ctx->addStatement(new StmtReturn(static_cast<Expression*>(texpr->detachValue())));

   p.simplify(3);
}


static void on_close_function( void * thing )
{
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   //printf( "Function closed\n" );
   /*
   SynFunc* func=ctx->currentFunc();
   TokenInstance* ti=new TokenInstance(0,0,sp.Expr);
   ti->setValue(func,func_deletor);

   //sp.simplify(0,ti);
   sp.addToStack(ti);
   */
}

static void apply_expr_func(const Rule& r,Parser& p)
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tf=sp.getNextToken();//T_function
   sp.getNextToken();// '('
   TokenInstance* targs=sp.getNextToken();
   sp.getNextToken();// ')'
   sp.getNextToken();// '\n'

   // todo: generate an anonymous name
   SynFunc* func=new SynFunc("anonymous",0,tf->line());
   NameList* list=static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti=new TokenInstance(tf->line(),tf->chr(),sp.Expr);
   Expression* expr=new ExprValue(Item(func));
   ti->setValue(expr,expr_deletor);

   // remove this stuff from the stack
   p.simplify(5,ti);

   // open a new main state for the function
   ctx->openFunc(func);
   p.pushState( "Main", on_close_function , &p );
}

//==========================================================
// SourceParser
//==========================================================



SourceParser::SourceParser():
   T_Openpar("("),
   T_Closepar(")"),
   T_OpenSquare("["),
   T_DotPar(".("),
   T_DotSquare(".["),
   T_CloseSquare("]"),
   T_OpenGraph("{"),
   T_CloseGraph("}"),

   T_Dot("."),
   T_Arrow("=>", 170 ),
   T_Comma( "," , 180 ),
   T_Cut("!"),

   T_UnaryMinus("(neg)",23),
   T_Power("**", 25),

   T_Times("*",30),
   T_Divide("/",30),
   T_Modulo("%",30),

   T_Plus("+",50),
   T_Minus("-",50),

   T_DblEq("==", 70),
   T_NotEq("!=", 70),
   T_Less("<", 70),
   T_Greater(">", 70),
   T_LE("<=", 70),
   T_GE(">=", 70),
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
   T_rule("rule"),

   T_while("while"),

   T_function("function"),
   T_return("return")
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

   S_Rule << "RULE"
      << (r_rule << "rule" << apply_rule << T_rule << T_EOL )
      ;

   S_Cut << "CUT"
      << (r_cut << "cut" << apply_cut << T_Cut << T_EOL )
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
      << (r_Expr_less << "Expr_less" << apply_expr_less << Expr << T_Less << Expr)
      << (r_Expr_greater << "Expr_greater" << apply_expr_greater << Expr << T_Greater << Expr)
      << (r_Expr_le << "Expr_le" << apply_expr_le << Expr << T_LE << Expr)
      << (r_Expr_ge << "Expr_ge" << apply_expr_ge << Expr << T_GE << Expr)
      << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)

      << (r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar )
      << (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare )
      << (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare )
      << (r_Expr_empty_dict << "Expr_empty_dict" << apply_expr_empty_dict << T_OpenSquare << T_Arrow << T_CloseSquare )
      << (r_Expr_array_decl << "Expr_array_decl" << apply_expr_array_decl << T_OpenSquare << ListExprOrPairs << T_CloseSquare )
      << (r_Expr_empty_dict2 << "Expr_empty_dict2" << apply_expr_empty_dict << T_DotSquare << T_Arrow << T_CloseSquare )
      << (r_Expr_array_decl2 << "Expr_array_decl2" << apply_expr_array_decl << T_DotSquare << SeqExprOrPairs << T_CloseSquare )
      << (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name)
      << (r_Expr_plus << "Expr_plus" << apply_expr_plus << Expr << T_Plus << Expr)
      << (r_Expr_minus << "Expr_minus" << apply_expr_minus << Expr << T_Minus << Expr)
      << (r_Expr_pars << "Expr_pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar)
      << (r_Expr_pars2 << "Expr_pars2" << apply_expr_pars << T_DotPar << Expr << T_Closepar)
      << (r_Expr_times << "Expr_times" << apply_expr_times << Expr << T_Times << Expr)
      << (r_Expr_div   << "Expr_div"   << apply_expr_div   << Expr << T_Divide << Expr )
      << (r_Expr_pow   << "Expr_pow"   << apply_expr_pow   << Expr << T_Power << Expr )
      // the lexer may find a non-unary minus when parsing it not after an operator...
      << (r_Expr_neg   << "Expr_neg"   << apply_expr_neg << T_Minus << Expr )
      // ... or find an unary minus when getting it after another operator.
      << (r_Expr_neg2   << "Expr_neg2"   << apply_expr_neg << T_UnaryMinus << Expr )
      << (r_Expr_Atom << "Expr_atom" << apply_expr_atom << Atom)
      << (r_Expr_function << "Expr_func" << apply_expr_func << T_function << T_Openpar << ListSymbol << T_Closepar << T_EOL)
      //<< (r_Expr_lambda << "Expr_lambda" << apply_expr_lambda << T_OpenGraph << ListSymbol << T_Arrow  )
      ;

   S_Function << "Function"
      /* This requires a bit of work << (r_function_short << "Function short" << apply_function_short
            << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar <<  T_Colon << Expr << T_EOL )
       */
      << (r_function << "Function decl" << apply_function
            << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
      ;

   S_Return << "Return"
      << (r_return << "return" << apply_return << T_return << Expr << T_EOL)
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

   ListExprOrPairs << "ListExprOrPairs"
      << (r_ListExprOrPairs_next_pair << "ListExprOrPairs_next_pair" << apply_ListExprOrPairs_next_pair << ListExprOrPairs << T_Comma << Expr << T_Arrow << Expr )
      << (r_ListExprOrPairs_next << "ListExprOrPairs_next" << apply_ListExprOrPairs_next << ListExprOrPairs << T_Comma << Expr )
      << (r_ListExprOrPairs_first_pair << "ListExprOrPairs_first_pair" << apply_ListExprOrPairs_first_pair << Expr << T_Arrow << Expr )
      << (r_ListExprOrPairs_first << "ListExprOrPairs_first" << apply_ListExprOrPairs_first << Expr )
      << (r_ListExprOrPairs_empty << "ListExprOrPairs_empty" << apply_ListExprOrPairs_empty )
      ;

   SeqExprOrPairs << "SeqExprOrPairs"
      << (r_SeqExprOrPairs_next_pair_cm << "SeqExprOrPairs_next_pair_cm" << apply_SeqExprOrPairs_next_pair_cm
            << SeqExprOrPairs << T_Comma << Expr << T_Arrow << Expr )
      << (r_SeqExprOrPairs_next_pair << "SeqExprOrPairs_next_pair" << apply_SeqExprOrPairs_next_pair
            << SeqExprOrPairs << Expr << T_Arrow << Expr )
      << (r_SeqExprOrPairs_next << "SeqExprOrPairs_next" << apply_SeqExprOrPairs_next << SeqExprOrPairs << Expr )
      << (r_SeqExprOrPairs_next_cm << "SeqExprOrPairs_next_cm" << apply_SeqExprOrPairs_next_cm << SeqExprOrPairs << T_Comma << Expr )
      << (r_SeqExprOrPairs_first_pair << "SeqExprOrPairs_first_pair" << apply_SeqExprOrPairs_first_pair << Expr << T_Arrow << Expr )
      << (r_SeqExprOrPairs_first << "SeqExprOrPairs_first" << apply_SeqExprOrPairs_first << Expr )
      << (r_SeqExprOrPairs_empty << "SeqExprOrPairs_empty" << apply_SeqExprOrPairs_empty )
      ;

   SeqExprOrPairs.prio(175);

   ListSymbol << "ListSymbol"
      << (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name )
      << (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name )
      << (r_ListSymbol_empty << "ListSymbol_empty" << apply_ListSymbol_empty )
      ;

   //==========================================================================
   //State declarations
   //
   s_Main << "Main"
      << S_Function
      << S_Autoexpr
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_Rule
      << S_Cut
      << S_End
      << S_Return
      ;


   addState( s_Main );
}

void SourceParser::onPushState()
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   pc->onStatePushed();
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

void SourceParser::reset()
{
   Parser::reset();
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
   pc->reset();
   pushState("Main");
}


}

/* end of sourceparser.cpp */
