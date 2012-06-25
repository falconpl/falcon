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
#include <falcon/symbol.h>
#include <falcon/error.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_function.h>

#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/stmtautoexpr.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/synclasses_id.h>

#include "private_types.h"
#include "falcon/psteps/exprclosure.h"
#include "falcon/psteps/exprsym.h"
#include "falcon/psteps/exprlit.h"
#include "falcon/psteps/stmtwhile.h"
#include "falcon/classes/classsyntree.h"

namespace Falcon {


// temporary statement used to keep track of the forming literal expression
class StmtTempLit: public Statement
{
public:
   SynTree* m_forming;

   StmtTempLit():
      Statement( 0,0 )
   {
      static Class* cls = Engine::instance()->syntreeClass();
      // don't record us, we're temp.
      m_forming = 0; // you can never know.
      m_discardable = true;
      handler(cls);
   }

   ~StmtTempLit()
   {
      delete m_forming;
   }
   
   virtual StmtTempLit* clone() const { return 0; }
};


using namespace Parsing;

static SynFunc* inner_apply_function( const Rule&, Parser& p, bool bHasExpr, bool isEta )
{
   //<< (r_Expr_function << "Expr_function" << apply_function << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   
   p.getNextToken();//T_function
   if( isEta ) p.getNextToken();// '*'
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
   Symbol* symfunc = 0; 
   // a method?
   if( ctx->currentClass() == 0 )
   {
      // if not, it's a global function
      symfunc = ctx->onGlobalDefined( *tname->asString(), alreadyDef );
      if( alreadyDef )
      {
         if( symfunc->type() != Symbol::e_st_extern )
         {
            // not free!
            p.addError( e_already_def,  p.currentSource(), tname->line(), tname->chr(), 0,
               String("at line ").N(symfunc->declaredAt()) );
            if( sp.interactive() )
            {
               p.simplify(tcount);
               return 0;
            }
         }
         else {
            symfunc->promoteToGlobal();
         }
         // otherwise, continue to compile the function.
      }
   }

   // Ok, we took the symbol.
   SynFunc* func = new SynFunc(*tname->asString(),0,tname->line());
   if( isEta ) func->setEta(true);
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
   inner_apply_function( r, p, false, false );
}

void apply_function_eta(const Rule& r,Parser& p)
{
   inner_apply_function( r, p, false, true );
}


void on_close_function( void* thing )
{
   // check if the function we have just created is a predicate.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func = ctx->currentFunc();
   if ( func->syntree().size() == 1 )
   {
      if( func->syntree().at(0)->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE )
      {
         func->setPredicate( true );
      }
   }
   
   // was this a closure?
   if( func->symbols().closedCount() > 0 ) {
      // change our token -- from function (value) to closure
      sp.getLastToken()->setValue( new ExprClosure(func), expr_deletor );
   }  
}

void on_close_lambda( void* thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func=ctx->currentFunc();

   int size = func->syntree().size();
   if ( size == 1 && func->syntree().at(0)->handler()->userFlags() == FALCON_SYNCLASS_ID_AUTOEXPR )
   {
      StmtAutoexpr* aexpr = static_cast<StmtAutoexpr*>( func->syntree().at(0) );
      Expression* evaluated = aexpr->detachExpr();
      StmtReturn* ret = new StmtReturn( evaluated );
      func->syntree().setNth(0, ret);
   }
   
   // was this a closure?
   if( func->symbols().closedCount() > 0 ) {
      // change our token -- from function (value) to closure
      sp.getLastToken()->setValue( new ExprClosure(func), expr_deletor );
   }  
}

void on_close_lit( void* thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempLit* lit = static_cast<StmtTempLit*>(ctx->currentStmt());

   ExprLit* elit = ctx->closeLitContext();
   SynTree* st = lit->m_forming;
   int size = st->size();
   if ( size == 1 && st->at(0)->handler()->userFlags() == FALCON_SYNCLASS_ID_AUTOEXPR )
   {
      StmtAutoexpr* aexpr = static_cast<StmtAutoexpr*>( st->at(0) );
      Expression* evaluated = aexpr->detachExpr();
      elit->setChild(evaluated);
   }
   else {
      lit->m_forming = 0;
      elit->setChild(st);
   }
   
   // The token was {~ (open lit expr), now we change it into an expression
   TokenInstance* ti = sp.getLastToken();
   ti->token(sp.Expr);
   ti->setValue( elit, expr_deletor );
   ctx->closeLitContext();
}


static void internal_expr_func(const Rule&, Parser& p, bool isEta )
{
   static Class* fcls = Engine::instance()->functionClass();
   
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tf = p.getNextToken();//T_function
   p.getNextToken();// '('
   if( isEta ) p.getNextToken();// '*'
   TokenInstance* targs = p.getNextToken();

   // todo: generate an anonymous name
   SynFunc* func = new SynFunc( "#anonymous", 0, tf->line() );
   if( isEta ) func->setEta(true);
   NameList* list=static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti= TokenInstance::alloc(tf->line(),tf->chr(), sp.Expr);

   // give the context the occasion to say something about this item
   Expression* expr= ctx->onStaticData( fcls, func );
   ti->setValue(expr,expr_deletor);

   // remove this stuff from the stack
   p.simplify(5,ti);

   // open a new main state for the function
   ctx->openFunc(func);
   // will check on close if the function is a predicate.
   p.pushState( "InlineFunc", on_close_function , &p );
}


void apply_expr_func(const Rule& r, Parser& p)
{
   internal_expr_func( r, p, false );   
}

void apply_expr_funcEta(const Rule& r, Parser& p)
{
   internal_expr_func( r, p, true );   
}


void apply_return_doubt(const Rule&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   p.getNextToken();//T_return
   p.getNextToken();//T_QMark
   TokenInstance* texpr=p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   StmtReturn* stmt_ret = new StmtReturn( expr, texpr->line(), texpr->chr() );
   stmt_ret->hasDoubt( true );
   ctx->addStatement(stmt_ret);

   p.simplify(4);
}

void apply_return_eval(const Rule&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   p.getNextToken();//T_return
   p.getNextToken();//T_Times
   TokenInstance* texpr=p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   StmtReturn* stmt_ret = new StmtReturn( expr, texpr->line(), texpr->chr() );
   stmt_ret->hasEval( true );
   ctx->addStatement(stmt_ret);

   p.simplify(4);
}

void apply_return_expr(const Rule&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   p.getNextToken();//T_return
   TokenInstance* texpr = p.getNextToken(); // Expr
   
   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   StmtReturn* stmt_ret = new StmtReturn( expr, texpr->line(), texpr->chr() );
   ctx->addStatement(stmt_ret);

   p.simplify(3);
}


void apply_return(const Rule&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   ctx->addStatement(new StmtReturn );

   p.simplify(2);
}


void apply_expr_lambda(const Rule&, Parser& p)
{
   // T_OpenGraph
   p.simplify(1);
   p.pushState( "LambdaStart", false );
}

void apply_expr_lit(const Rule&, Parser& p)
{
   // T_OpenLit
   
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   // keep the token; we'll be converting it into the exprlit when done.
   TokenInstance* ti = p.getNextToken();
   StmtTempLit* tlit = new StmtTempLit;
   tlit->m_forming = new SynTree(ti->line(), ti->chr());

   ctx->openBlock( tlit, tlit->m_forming );
   ExprLit* lit = new ExprLit(tlit->line(),tlit->chr());
   ctx->openLitContext(lit);
   p.pushState( "InlineFunc", on_close_lit , &p );
}

void apply_expr_parametric_lit(const Rule&, Parser& p)
{
   // T_OpenLit
   p.simplify(1);
   p.pushState( "LambdaStart", false );
}

static void internal_lambda_params(const Rule&, Parser& p, bool isEta )
{
   static Class* fcls = Engine::instance()->functionClass();
   // ListSymbol << T_Arrow
   SourceParser& sp = static_cast<SourceParser&>(p);

   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* lsym = sp.getNextToken();


   // and add the function state.
   SynFunc* func=new SynFunc("#anonymous", 0, lsym->line());
   if( isEta ) func->setEta(true);
   NameList* list=static_cast<NameList*>(lsym->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti = TokenInstance::alloc(lsym->line(),lsym->chr(), sp.Expr);
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

void apply_lambda_params(const Rule& r, Parser& p)
{
   internal_lambda_params( r, p, false );
}

void apply_lambda_params_eta(const Rule& r, Parser& p)
{
   internal_lambda_params( r, p, true );
}

}

/* end of parser_function.cpp */
