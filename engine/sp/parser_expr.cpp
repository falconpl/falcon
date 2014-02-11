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
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprincdec.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprcompare.h>
#include <falcon/psteps/exprmath.h>
#include <falcon/psteps/exprin.h>
#include <falcon/psteps/exprinvoke.h>
#include <falcon/psteps/exprnotin.h>
#include <falcon/psteps/exprdot.h>
#include <falcon/psteps/expreeq.h>
#include <falcon/psteps/exprneg.h>
#include <falcon/psteps/exprbitwise.h>
#include <falcon/psteps/exproob.h>
#include <falcon/psteps/exprlogic.h>
#include <falcon/psteps/exprlit.h>
#include <falcon/psteps/exprunquote.h>
#include <falcon/psteps/exprevalret.h>
#include <falcon/psteps/exprevalretexec.h>
#include <falcon/psteps/exprstripol.h>
#include <falcon/psteps/exprprovides.h>
#include <falcon/psteps/exprnamed.h>

#include <falcon/psteps/exprcompose.h>

#include "falcon/parser/lexer.h"

namespace Falcon {

using namespace Parsing;

bool expr_errhand(const NonTerminal&, Parser& p, int)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = p.getLastToken();
   p.addError( e_syn_expr, p.currentSource(), ti2->line(), ti2->chr(), ti->line() );
   
   p.setErrorMode(&p.T_EOL);
   return true;
}

//=======================================================
// Standard binary expressions
//

static void apply_expr_binary( const NonTerminal&, Parser& p, BinaryExpression* bexpr )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   bexpr->decl(v1->line(), v1->chr());
   TokenInstance* ti = TokenInstance::alloc(v1->line(), v1->chr(), sp.Expr);
   bexpr->first(static_cast<Expression*>(v1->detachValue()));
   bexpr->second(static_cast<Expression*>(v2->detachValue()));
   ti->setValue( bexpr, treestep_deletor );

   p.simplify(3,ti);
}


void apply_expr_equal( const NonTerminal& r, Parser& p )
{
   // << (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr)
   apply_expr_binary(r, p, new ExprEQ );
}

 void apply_expr_diff( const NonTerminal& r, Parser& p )
{
   // << (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr)
   apply_expr_binary(r, p, new ExprNE );
}

 void apply_expr_less( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLT );
}

 void apply_expr_greater( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGT );
}

 void apply_expr_le( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLE );
}

 void apply_expr_ge( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprGE );
}


 void apply_expr_eeq( const NonTerminal& r, Parser& p )
{
  // << (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr)
  apply_expr_binary( r, p, new ExprEEQ );
}

void apply_expr_in( const NonTerminal& r, Parser& p )
{
  apply_expr_binary( r, p, new ExprIn );
}

void apply_expr_notin( const NonTerminal& r, Parser& p )
{
  apply_expr_binary( r, p, new ExprNotin );
}

void apply_expr_plus( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPlus );
}

void apply_expr_preinc( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* plpl  = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc( plpl->line(), plpl->chr(), sp.Expr );
   ti2->setValue( new ExprPreInc(static_cast<Expression*>(value->detachValue()), plpl->line(), plpl->chr()), treestep_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_postinc( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* value = p.getNextToken();
   /*TokenInstance* plpl  =*/ p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprPostInc(static_cast<Expression*>(value->detachValue()), value->line(), value->chr()), treestep_deletor );

   p.simplify(2,ti2);
}
 
 void apply_expr_predec( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* plpl  = p.getNextToken();
   TokenInstance* value = p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc( plpl->line(), plpl->chr(), sp.Expr );
   ti2->setValue( new ExprPreDec(static_cast<Expression*>(value->detachValue()), plpl->line(), plpl->chr()), treestep_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_postdec( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* value = p.getNextToken();
   /*TokenInstance* plpl  =*/ p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc( value->line(), value->chr(), sp.Expr );
   ti2->setValue( new ExprPostDec(static_cast<Expression*>(value->detachValue()), value->line(), value->chr()), treestep_deletor );

   p.simplify(2,ti2);
}

 void apply_expr_minus( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprMinus );
}

 void apply_expr_times( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprTimes );
}

void apply_expr_div( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprDiv );
}

void apply_expr_mod( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprMod );
}

void apply_expr_pow( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprPow );
}

void apply_expr_shr( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprRShift );
}

void apply_expr_shl( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprLShift );
}

void apply_expr_and( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprAnd );
}

void apply_expr_or( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprOr );
}

void apply_expr_band( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprBAND );
}

void apply_expr_bor( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprBOR );
}

void apply_expr_bxor( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprBXOR );
}

void apply_expr_invoke( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprInvoke );
}

void apply_expr_compose( const NonTerminal& r, Parser& p )
{
   apply_expr_binary(r, p, new ExprCompose );
}

//==================================================================
// Auto expressions
//

void apply_expr_auto( const NonTerminal&, Parser& p, BinaryExpression* aexpr )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* tfirst = p.getNextToken();
   TokenInstance* oper = p.getNextToken();
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
   aexpr->decl( oper->line(), oper->chr() );
   aexpr->first( firstPart );
   aexpr->second( secondPart );
   TokenInstance* ti = TokenInstance::alloc(tfirst->line(), tfirst->chr(), sp.Expr);
   ti->setValue( aexpr, treestep_deletor );
   p.simplify( 3, ti );
}


void apply_expr_auto_add( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoPlus;
   apply_expr_auto( r, p, aexpr );
}


void apply_expr_auto_sub( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoMinus;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_times( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoTimes;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_div( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoDiv;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_mod( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoMod;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_pow( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoPow;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_shr( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoRShift;
   apply_expr_auto( r, p, aexpr );
}

void apply_expr_auto_shl( const NonTerminal&r, Parser& p )
{
   BinaryExpression* aexpr = new ExprAutoLShift;
   apply_expr_auto( r, p, aexpr );
}

      
//=======================================================
// Unary expressions
//

static void apply_expr_unary( const NonTerminal&, Parser& p, UnaryExpression* un )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   
   TokenInstance* oper = p.getNextToken();
   TokenInstance* value = p.getNextToken();  // expr   
   
   // get the expression coming from the value and unarize it.
   Expression* other = static_cast<Expression*>(value->detachValue());
   un->decl(oper->line(), oper->chr());
   un->first( other );
   
   // Complexify the value
   value->token( sp.Expr );   // be sure to change it
   value->setValue( un, treestep_deletor );
   p.simplify(1); // remove 1 token and keep the value.
}


void apply_expr_neg( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprNeg );
}

void apply_expr_not( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprNot );
}

void apply_expr_bnot( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprBNOT );
}

void apply_expr_oob( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprOob );
}

void apply_expr_deoob( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprDeoob );
}

void apply_expr_xoob( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprXorOob );
}

void apply_expr_isoob( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprIsOob );
}

void apply_expr_str_ipol( const NonTerminal& r, Parser& p )
{
   apply_expr_unary( r, p, new ExprStrIPol );
}

void apply_expr_unquote( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);   
   TokenInstance* ti = p.getNextToken();
   TokenInstance* value = p.getNextToken();  // symbol 
   
   Expression* child = static_cast<Expression*>(value->detachValue());
   ExprUnquote* un = new ExprUnquote( child , value->line(), value->chr());
   // Complexify the value
   value->token( sp.Expr );   // be sure to change it
   value->setValue( un, treestep_deletor );
   p.simplify(1); // remove 1 token and keep the value.

   // now, was it ok?
   ParserContext* ctx = static_cast<ParserContext*>(p.context()); 
   ExprLit* lit = ctx->currentLitContext();
   if( lit == 0 ) {
      p.addError(e_enc_fail, p.currentSource(), ti->line(), ti->chr(), 0);
   }
   else {
      lit->registerUnquote(child);
   }

}

void apply_expr_evalret( const NonTerminal&r, Parser& p )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   if( ctx->currentLitContext() == 0 ) {
      p.addError(e_syn_evalret_out, p.currentSource(), p.currentLine(), p.currentLexer()->character(), 0);
   }
   
   // add the expression nevertheless.
   apply_expr_unary( r, p, new ExprEvalRet );
   
}

void apply_expr_evalret_exec( const NonTerminal&r, Parser& p )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   if( ctx->currentLitContext() == 0 ) {
      p.addError(e_syn_evalret_out, p.currentSource(), p.currentLine(), p.currentLexer()->character(), 0);
   }
   
   // add the expression nevertheless.
   apply_expr_unary( r, p, new ExprEvalRetExec );
}


void apply_expr_evalret_doubt( const NonTerminal&r, Parser& p )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   if( ctx->currentLitContext() == 0 ) {
      p.addError(e_syn_evalret_out, p.currentSource(), p.currentLine(), p.currentLexer()->character(), 0);
   }

   // add the expression nevertheless.
   apply_expr_unary( r, p, new ExprEvalRetDoubt );
}

//=======================================================
// Other expressions.
//

void apply_expr_pars( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Expr);
   ti2->setValue( ti->detachValue(), treestep_deletor );
   p.simplify(3,ti2);
}

void apply_expr_dot( const NonTerminal&, Parser& p )
{
   // << (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();
   TokenInstance* v2 = p.getNextToken();

   TokenInstance* ti = TokenInstance::alloc(v1->line(), v1->chr(), sp.Expr);
   ti->setValue( new ExprDot(
         *v2->asString(),
         static_cast<Expression*>(v1->detachValue())
      ), treestep_deletor );

   p.simplify(3,ti);
}


void apply_expr_provides( const NonTerminal& , Parser& p )
{
   // << Expr << T_Provides << T_Name
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* texpr = p.getNextToken();   // expr
   TokenInstance* toper = p.getNextToken();
   TokenInstance* tprop = p.getNextToken();  // name

   // get the expression coming from the value and unarize it.
   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   const String& prop = *tprop->asString();
   ExprProvides* provides = new ExprProvides( expr, prop, toper->line(), toper->chr() );


   // Complexify the value
   texpr->token( sp.Expr );   // be sure to change it
   texpr->setValue( provides, treestep_deletor );
   p.trim(2); // remove the 2 tokens in front
}


void apply_expr_named( const NonTerminal& , Parser& p )
{
   // << T_Name << T_Disjunct << T_Expr
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();   // name
   p.getNextToken();                          // disjunct
   TokenInstance* texpr = p.getNextToken();   // expression
   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   TokenInstance* ti = TokenInstance::alloc(tname->line(), tname->chr(), sp.Expr);
   Expression* named = new ExprNamed( *tname->asString(), expr, tname->line(), tname->chr() );
   ti->setValue( named, treestep_deletor );

   p.simplify(3, ti); // remove the 2 tokens in front
}

}

/* end of parser_expr.cpp */
