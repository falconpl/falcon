/*
   FALCON - The Falcon Programming Language.
   FILE: parser_expr.cpp

   Falcon source parser -- expression parsing handlers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_expr.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_expr.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/exprincdec.h>
#include <falcon/exprvalue.h>
#include <falcon/exprcompare.h>
#include <falcon/exprmath.h>
#include <falcon/exprdot.h>

namespace Falcon {

using namespace Parsing;

bool expr_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = p.getLastToken();
   p.addError( e_syn_expr, p.currentSource(), ti2->line(), ti2->chr(), ti->line() );

   // remove the whole expression.
   p.resetNextToken();
   p.simplify( p.availTokens() );
   return true;
}

//=======================================================
// Standard binary expressions
//

static void apply_expr_binary( const Rule&, Parser& p, BinaryExpression* bexpr )
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


void apply_expr_equal( const Rule& r, Parser& p )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   apply_expr_binary(r, p, new ExprEQ );
}

 void apply_expr_diff( const Rule& r, Parser& p )
{
   // << (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr)
   apply_expr_binary(r, p, new ExprNE );
}

 void apply_expr_less( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLT );
}

 void apply_expr_greater( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGT );
}

 void apply_expr_le( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLE );
}

 void apply_expr_ge( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGE );
}


 void apply_expr_eeq( const Rule& r, Parser& p )
{
  // << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)
  apply_expr_binary( r, p, new ExprEEQ );
}


void apply_expr_plus( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPlus );
}

void apply_expr_preinc(const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* plpl  = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( plpl->line(), plpl->chr(), sp.Expr );
   ti2->setValue( new ExprPreInc(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_postinc(const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* value = p.getNextToken();
   /*TokenInstance* plpl  =*/ p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprPostInc(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}
 
 void apply_expr_predec(const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* plpl  = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( plpl->line(), plpl->chr(), sp.Expr );
   ti2->setValue( new ExprPreDec(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_postdec(const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* value = p.getNextToken();
   /*TokenInstance* plpl  =*/ p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprPostDec(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_minus( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprMinus );
}

 void apply_expr_times( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprTimes );
}

 void apply_expr_div( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprDiv );
}


void apply_expr_pow( const Rule& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPow );
}
 

void apply_expr_auto( const Rule&, Parser& p, BinaryExpression* aexpr )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* tfirst = p.getNextToken();
   (void) p.getNextToken();
   TokenInstance* tsecond = p.getNextToken();
   
   Expression* firstPart = static_cast<Expression*>(tfirst->detachValue());
   // assignable expressions are only expressions having a lvalue pstep:
   // -- symbols
   // -- accessors
   if( firstPart->lvalueStep() == 0  )
   {
      p.addError( e_assign_sym, p.currentSource(), tfirst->line(), tfirst->chr(), 0 );
   }
   
   Expression* secondPart = static_cast<Expression*>(tsecond->detachValue());
   aexpr->first( firstPart );
   aexpr->second( secondPart );
   TokenInstance* ti = new TokenInstance(tfirst->line(), tfirst->chr(), sp.Expr);
   ti->setValue( aexpr, expr_deletor );
   p.simplify( 3, ti );
}


void apply_expr_auto_add( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoPlus;
   apply_expr_auto( r, p, aexpr );
}


void apply_expr_auto_sub( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoMinus;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_times( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoTimes;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_div( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoDiv;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_mod( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoMod;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_pow( const Rule&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoPow;
   apply_expr_auto( r, p, aexpr );
}


 void apply_expr_neg( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   (void) p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = new TokenInstance( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprNeg(static_cast<Expression*>(value->detachValue())), expr_deletor );

   p.simplify(2,ti2);
}


void apply_expr_pars( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti2 = new TokenInstance(ti->line(), ti->chr(), sp.Expr);
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(3,ti2);
}

void apply_expr_dot( const Rule&, Parser& p )
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

}

/* end of parser_expr.cpp */
