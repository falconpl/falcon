/*
   FALCON - The Falcon Programming Language.
   FILE: parser_for.cpp

   Parser for Falcon source files -- for statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Aug 2011 16:56:41 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parsercontext.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>
#include <falcon/error.h>
#include <falcon/expression.h>
#include <falcon/statement.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_for.h>

#include <falcon/psteps/stmtfor.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/synclasses_id.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

bool for_errhand(const NonTerminal&, Parser& p, int)
{
   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = p.getLastToken();

   if( p.lastErrorLine() < ti->line() )
   {
      // generate another error only if we didn't notice it already.
      p.addError( e_syn_for, p.currentSource(), ti->line(), ti2->chr() );
   }

   if( ! p.interactive() )
   {
      // put in a fake if statement (else, subsequent else/elif/end would also cause errors).
      SynTree* fakeTree = new SynTree;
      StmtForIn* fakeFor = new StmtForIn( new ExprValue(1), ti->line(), ti->chr() );
      fakeFor->body( fakeTree );
      ParserContext* st = static_cast<ParserContext*>(p.context());
      st->openBlock( fakeFor, fakeTree );
   }
   else
   {
      MESSAGE2( "for_errhand -- Ignoring FOR in interactive mode." );
   }
   // on interactive parsers, let the whole instruction to be destroyed.

   p.setErrorMode(&p.T_EOF);
   return true;
}


static StmtForTo* internal_for_to( Parser& p, TokenInstance* tfor,
         const String& name, Expression* start, Expression* end, Expression* step )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   ctx->defineSymbol( name );
   if( start != 0 ) ctx->accessSymbols( start );
   if( end != 0 ) ctx->accessSymbols( end );
   if( step != 0 ) ctx->accessSymbols( step );

   /*
   int64 iStart = 0;
   int64 iEnd = 0;
   int64 iStep = 0;
   
   if( start->type() == Expression::t_value )
   {
      ExprValue* vStart = static_cast<ExprValue*>(start);
      if( vStart->item().isOrdinal() )
      {
         iStart = vStart->item().forceInteger();
         delete start;
         start = 0;
      }
   }
   
   if( end->type() == Expression::t_value )
   {
      ExprValue* vEnd = static_cast<ExprValue*>(end);
      if( vEnd->item().isOrdinal() )
      {
         iEnd = vEnd->item().forceInteger();
         delete end;
         end = 0;
      }
   }
   
   if( step != 0 && step->type() == Expression::t_value )
   {
      ExprValue* vStep = static_cast<ExprValue*>(step);
      if( vStep->item().isOrdinal() )
      {
         iStep = vStep->item().forceInteger();
         delete step;
         step = 0;
      }
   }
   
   StmtForTo* ft = new StmtForTo( sym, iStart, iEnd, iStep, tfor->line(), tfor->chr() );
      
   if( start != 0 ) ft->startExpr( start );
   if( end != 0 ) ft->endExpr( end );
   if( step != 0 ) ft->stepExpr( step );
   */
   
   StmtForTo* ft = new StmtForTo( Engine::getSymbol(name),
               start, end, step, tfor->line(), tfor->chr() );
   return ft;
}


void apply_for_to_step( const NonTerminal&, Parser& p )
{
   // << T_for << Symbol << T_EqSign << Expr << T_to << Expr << T_comma << Expr << T_EOL )
   ParserContext* st = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tsym = p.getNextToken();
   p.getNextToken(); // =
   TokenInstance* texpr_start = p.getNextToken();
   p.getNextToken(); // to
   TokenInstance* texpr_end = p.getNextToken();
   p.getNextToken(); // ,
   TokenInstance* texpr_step = p.getNextToken();
   
   Expression* expr_start = static_cast<Expression*>(texpr_start->detachValue());
   Expression* expr_end = static_cast<Expression*>(texpr_end->detachValue());
   Expression* expr_step = static_cast<Expression*>(texpr_step->detachValue());

   StmtForTo* ft = internal_for_to( p, tfor, *tsym->asString(), expr_start, expr_end, expr_step );
   SynTree* body = new SynTree;
   ft->body( body );
   
   st->openBlock( ft, body );

   // clear the stack
   p.simplify(9);
}


void apply_for_to_step_short( const NonTerminal&, Parser& p )
{
   // << T_for << Symbol << T_EqSign << Expr << T_to << Expr << T_comma << Expr << T_COLON )
   ParserContext* st = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tsym = p.getNextToken();
   p.getNextToken(); // =
   TokenInstance* texpr_start = p.getNextToken();
   p.getNextToken(); // to
   TokenInstance* texpr_end = p.getNextToken();
   p.getNextToken(); // ,
   TokenInstance* texpr_step = p.getNextToken();
   
   Expression* expr_start = static_cast<Expression*>(texpr_start->detachValue());
   Expression* expr_end = static_cast<Expression*>(texpr_end->detachValue());
   Expression* expr_step = static_cast<Expression*>(texpr_step->detachValue());

   StmtForTo* ft = internal_for_to( p, tfor, *tsym->asString(), expr_start, expr_end, expr_step );
   SynTree* body = new SynTree;
   ft->body( body );
   
   st->openBlock( ft, body, true );

   // clear the stack
   p.simplify(9);
}

void apply_for_to( const NonTerminal&, Parser& p )
{
   // << T_for << Symbol << T_EqSign << Expr << T_to << Expr << T_EOL )
   ParserContext* st = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tsym = p.getNextToken();
   p.getNextToken(); // =
   TokenInstance* texpr_start = p.getNextToken();
   p.getNextToken(); // to
   TokenInstance* texpr_end = p.getNextToken();
   
   Expression* expr_start = static_cast<Expression*>(texpr_start->detachValue());
   Expression* expr_end = static_cast<Expression*>(texpr_end->detachValue());

   StmtForTo* ft = internal_for_to( p, tfor, *tsym->asString(), expr_start, expr_end, 0);
   SynTree* body = new SynTree;
   ft->body( body );
   
   st->openBlock( ft, body );

   // clear the stack
   p.simplify(7);
}


void apply_for_to_short( const NonTerminal&, Parser& p )
{
   // << T_for << Symbol << T_EqSign << Expr << T_to << Expr << T_EOL )
   ParserContext* st = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tsym = p.getNextToken();
   p.getNextToken(); // =
   TokenInstance* texpr_start = p.getNextToken();
   p.getNextToken(); // to
   TokenInstance* texpr_end = p.getNextToken();
   
   Expression* expr_start = static_cast<Expression*>(texpr_start->detachValue());
   Expression* expr_end = static_cast<Expression*>(texpr_end->detachValue());

   StmtForTo* ft = internal_for_to( p, tfor, *tsym->asString(), expr_start, expr_end, 0);
   SynTree* body = new SynTree;
   ft->body( body );
   
   st->openBlock( ft, body, true );

   // clear the stack
   p.simplify(7);
}


void apply_for_in( const NonTerminal&, Parser& p )
{
   // << T_for << NeListSymbol << T_in << Expr << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tlist = p.getNextToken();   
   p.getNextToken(); // in
   TokenInstance* texpr = p.getNextToken();
   
   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   NameList* list = static_cast<NameList*>(tlist->detachValue());
   NameList::iterator iter = list->begin();
   
   StmtForIn* forin = new StmtForIn( expr, tfor->line(), tfor->chr() );
   
   while( iter != list->end() )
   {
      const String& name = *iter;
      ctx->defineSymbol( name );
      forin->addParameter( Engine::getSymbol(name  ) );
      ++iter;
   }
         
   SynTree* body = new SynTree;
   forin->body( body );
   ctx->openBlock( forin, body );
   
   // clear the stack
   p.simplify(5);
}

void apply_for_in_short( const NonTerminal&, Parser& p )
{
   // << T_for << NeListSymbol << T_in << Expr << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tfor = p.getNextToken();
   TokenInstance* tlist = p.getNextToken();   
   p.getNextToken(); // in
   TokenInstance* texpr = p.getNextToken();
   
   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   NameList* list = static_cast<NameList*>(tlist->detachValue());
   NameList::iterator iter = list->begin();
   
   StmtForIn* forin = new StmtForIn( expr, tfor->line(), tfor->chr() );
   
   while( iter != list->end() )
   {
      const String& name = *iter;
      ctx->defineSymbol( name );
      forin->addParameter( Engine::getSymbol(name) );
      ++iter;
   }
         
   SynTree* body = new SynTree;
   forin->body( body );
   ctx->openBlock( forin, body, true );
   
   // clear the stack
   p.simplify(5);
}


static void apply_forfirst_internal( const NonTerminal&, Parser& p, bool bShort )
{
   // << T_forfirst << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   TreeStep* stmt = ctx->currentStmt();
   
   // TODO: Use a typing system?
   if( stmt == 0 || stmt->handler()->userFlags() != FALCON_SYNCLASS_ID_FORCLASSES )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_forfirst_outside, p.currentSource(), ti->line(), ti->chr() );
   }
   
   StmtForBase* base = static_cast<StmtForBase*>( stmt );
   if( base->forFirst() != 0 )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_already_forfirst, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // anyhow, add the block
   SynTree* body = new SynTree;
   base->forFirst( body );
   ctx->openBlock( new StmtTempFor, body, bShort );
   
   // clear the stack
   p.simplify(2);
}


void apply_forfirst( const NonTerminal& r, Parser& p )
{
   apply_forfirst_internal( r, p, false );
}


void apply_forfirst_short( const NonTerminal& r, Parser& p )
{
   apply_forfirst_internal( r, p, true );
}


static void apply_formiddle_internal( const NonTerminal&, Parser& p, bool isShort )
{
   // << T_formiddle << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   TreeStep* stmt = ctx->currentStmt();
   
   if( stmt == 0 || stmt->handler()->userFlags() != FALCON_SYNCLASS_ID_FORCLASSES )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_formiddle_outside, p.currentSource(), ti->line(), ti->chr() );
   }
   
   StmtForBase* base = static_cast<StmtForBase*>( stmt );
   if( base->forMiddle() != 0 )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_already_formiddle, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // anyhow, add the block
   SynTree* body = new SynTree;
   base->forMiddle( body );
   ctx->openBlock( new StmtTempFor, body, isShort );
   
   // clear the stack
   p.simplify(2);
}


void apply_formiddle( const NonTerminal& r, Parser& p )
{
   apply_formiddle_internal( r, p, false );
}


void apply_formiddle_short( const NonTerminal& r, Parser& p )
{
   apply_formiddle_internal( r, p, true );
}

static void apply_forlast_internal( const NonTerminal&, Parser& p, bool isShort )
{
   // << T_forlast << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());   
   TreeStep* stmt = ctx->currentStmt();
   
   if( stmt == 0 || stmt->handler()->userFlags() != FALCON_SYNCLASS_ID_FORCLASSES )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_forlast_outside, p.currentSource(), ti->line(), ti->chr() );
   }
   
   StmtForBase* base = static_cast<StmtForBase*>( stmt );
   if( base->forLast() != 0 )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_already_forlast, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // anyhow, add the block
   SynTree* body = new SynTree;
   base->forLast( body );
   ctx->openBlock( new StmtTempFor, body, isShort );
   
   // clear the stack
   p.simplify(2);
}


void apply_forlast( const NonTerminal& r, Parser& p )
{
   apply_forlast_internal( r, p, false );
}

void apply_forlast_short( const NonTerminal& r, Parser& p )
{
   apply_forlast_internal( r, p, true );
}

}

/* end of parser_for.cpp */
