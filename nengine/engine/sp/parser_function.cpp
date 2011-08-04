/*
   FALCON - The Falcon Programming Language.
   FILE: parser_function.cpp

   Parser for Falcon source files -- function declarations handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_function.cpp"

#include <falcon/setup.h>
#include <falcon/statement.h>
#include <falcon/stmtautoexpr.h>

#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/globalsymbol.h>
#include <falcon/error.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_function.h>


#include "private_types.h"

namespace Falcon {

using namespace Parsing;

static SynFunc* inner_apply_function( const Rule&, Parser& p, bool bHasExpr )
{
   //<< (r_Expr_function << "Expr_function" << apply_function << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   p.getNextToken();//T_function
   TokenInstance* tname = p.getNextToken();
   p.getNextToken();// '('
   TokenInstance* targs = p.getNextToken();

   TokenInstance* tstatement = 0;
   int tcount = bHasExpr ? 8 : 6;

   if( bHasExpr )
   {
      tstatement = p.getNextToken();
   }

   // Are we already in a function?
   if( ctx->currentFunc() != 0 || ctx->currentStmt() != 0)
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
   SynFunc* func = new SynFunc(*tname->asString(),0,tname->line());
   NameList* list = static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   if ( bHasExpr )
   {
      Expression* sa = static_cast<Expression*>(tstatement->detachValue());
      func->syntree().append(new StmtReturn(sa));
      p.simplify(tcount);
   }
   else
   {
      ctx->openFunc(func, symfunc);
      p.simplify(tcount);
      p.pushState( "Main" );
   }

   return func;
}

void apply_function(const Rule& r,Parser& p)
{
   inner_apply_function( r, p, false );
}

/*
//TODO Remove
static void apply_function_short(const Rule& r, Parser& p)
{
   inner_apply_function( r, p, true );
}
*/


void on_close_function( void* )
{
   // TODO: name the function
   /*
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func=ctx->currentFunc();
   */
}

void on_close_lambda( void* thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func=ctx->currentFunc();

   int size = func->syntree().size();
   if ( size == 1 && func->syntree().at(0)->type() == Statement::autoexpr_t )
   {
      StmtAutoexpr* aexpr = static_cast<StmtAutoexpr*>( func->syntree().at(0) );
      StmtReturn* ret = new StmtReturn( aexpr->detachExpr() );
      func->syntree().set(0, ret);
   }
}


void apply_expr_func(const Rule&, Parser& p)
{
   static Class* fcls = Engine::instance()->functionClass();
   
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tf = p.getNextToken();//T_function
   p.getNextToken();// '('
   TokenInstance* targs = p.getNextToken();

   // todo: generate an anonymous name
   SynFunc* func=new SynFunc( "anonymous", 0, tf->line() );
   NameList* list=static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti=new TokenInstance(tf->line(),tf->chr(), sp.Expr);

   // give the context the occasion to say something about this item
   Expression* expr= ctx->onStaticData( fcls, func );
   ti->setValue(expr,expr_deletor);

   // remove this stuff from the stack
   p.simplify(5,ti);

   // open a new main state for the function
   ctx->openFunc(func);
   p.pushState( "InlineFunc", on_close_function , &p );
}



void apply_return(const Rule&, Parser& p)
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   sp.getNextToken();//T_function
   TokenInstance* texpr=sp.getNextToken();
   sp.getNextToken();//T_EOL

   ctx->addStatement(new StmtReturn(static_cast<Expression*>(texpr->detachValue())));

   p.simplify(3);
}


void apply_expr_lambda(const Rule&, Parser& p)
{
   // T_OpenGraph
   p.simplify(1);
   p.pushState( "LambdaStart", false );
}


void apply_lambda_params(const Rule&, Parser& p)
{
   static Class* fcls = Engine::instance()->functionClass();
   // ListSymbol << T_Arrow
   SourceParser& sp = static_cast<SourceParser&>(p);

   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* lsym = sp.getNextToken();


   // and add the function state.
   SynFunc* func=new SynFunc("anonymous", 0, lsym->line());
   NameList* list=static_cast<NameList*>(lsym->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti = new TokenInstance(lsym->line(),lsym->chr(), sp.Expr);
   Expression* expr = ctx->onStaticData( fcls, func );   
   ti->setValue(expr,expr_deletor);

   // remove this stuff from the stack
   p.simplify(2,ti);
   // remove the lambdastart state
   p.popState();

   // open a new main state for the function
   ctx->openFunc(func);
   p.pushState( "InlineFunc", on_close_lambda , &p );
}



}

/* end of parser_function.cpp */
