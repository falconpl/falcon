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

#include <falcon/parser/parser.h>
#include <falcon/psteps/exprtree.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_function.h>

#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/synclasses_id.h>

#include "private_types.h"
#include <falcon/psteps/exprclosure.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprlit.h>
#include <falcon/psteps/stmtwhile.h>
#include <falcon/classes/classsyntree.h>
#include <falcon/psteps/exprep.h>
#include <falcon/stdhandlers.h>


namespace Falcon {


// temporary statement used to keep track of the forming literal expression
class StmtTempLit: public Statement
{
public:
   SynTree* m_forming;

   StmtTempLit():
      Statement( 0,0 )
   {
      static Class* cls = Engine::handlers()->syntreeClass();
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
   virtual void render( TextWriter* , int32 ) const {};
};

using namespace Parsing;

static SynFunc* inner_apply_function( const NonTerminal&, Parser& p, bool bHasExpr, bool isEta, bool isStatic )
{
   //<< (r_Expr_function << "Expr_function" << apply_function << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   //SourceParser& sp = *static_cast<SourceParser*>(&p);
   
   if( isStatic ) p.getNextToken(); // 'static'
   p.getNextToken();//T_function
   if( isEta ) p.getNextToken();// '*'
   TokenInstance* tname = p.getNextToken();
   p.getNextToken();// '('
   TokenInstance* targs = p.getNextToken();

   TokenInstance* tstatement = 0;
   int tcount = bHasExpr ? 8 : 6;
   if (isStatic)
   {
      tcount++;
   }

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
      func->syntree().append(new StmtReturn(sa, tstatement->line(), tstatement->chr()));
      p.simplify(tcount);
   }
   else
   {
      p.simplify(tcount);

      // don't add the mantra if we're in a class.
      if( ctx->currentClass() != 0 )
      {
         ctx->onOpenMethod( (Class*)ctx->currentClass(), func );
         ctx->openFunc(func, isStatic );
         p.pushState( "Main" );
      }
      // try to create the function
      else if( ctx->onOpenFunc( func ) != 0 || ! p.interactive() ) {
         // non-interactive compiler must go on even on error.
         ctx->openFunc(func);
         p.pushState( "Main" );
      }
   }

   return func;
}

void apply_function( const NonTerminal& r,Parser& p)
{
   inner_apply_function( r, p, false, false, false );
}

void apply_function_eta( const NonTerminal& r,Parser& p)
{
   inner_apply_function( r, p, false, true, false );
}

void apply_static_function( const NonTerminal& r,Parser& p)
{
   inner_apply_function( r, p, false, false, true );
}

void apply_static_function_eta( const NonTerminal& r,Parser& p)
{
   inner_apply_function( r, p, false, false, true );
}

void on_close_function( void* thing )
{
   // check if the function we have just created is a predicate.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func = ctx->currentFunc();
   
   // was this a closure?
   if( func->hasClosure() ) {
      // change our token -- from function (value) to closure
      sp.getLastToken()->setValue( new ExprClosure(func), treestep_deletor );
   }  
}

void on_close_lambda( void* thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   SynFunc* func=ctx->currentFunc();

   int size = func->syntree().size();
   if ( size == 1 && func->syntree().at(0)->category() == TreeStep::e_cat_expression )
   {
      Expression* evaluated = static_cast<Expression*>( func->syntree().detach(0) );
      StmtReturn* ret = new StmtReturn( evaluated, evaluated->line(), evaluated->chr() );
      func->syntree().append(ret);
   }
   
   // was this a closure?
   if( func->hasClosure() ) {
      // change our token -- from function (value) to closure
      sp.getLastToken()->setValue( new ExprClosure(func), treestep_deletor );
   }  
}

void on_close_lit( void* thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempLit* lit = static_cast<StmtTempLit*>(ctx->currentStmt());

   ExprLit* elit = ctx->closeLitContext();
   ExprTree* et = static_cast<ExprTree*>(elit->child());
   SynTree* st = lit->m_forming;
   int size = st->size();
   if ( size == 1 )
   {
      Expression* evaluated = static_cast<Expression*>( st->detach(0) );
      if( et == 0 ) {
         elit->setChild(evaluated);
      }
      else {
         et->setChild(evaluated);
      }
   }
   else {
      lit->m_forming = 0;
      if( et == 0 ) {
         elit->setChild(st);
      }
      else {
         et->setChild(st);
      }
   }
}


static void internal_expr_func( const NonTerminal&, Parser& p, bool isEta )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tf = p.getNextToken();//T_function
   p.getNextToken();// '('
   if( isEta ) p.getNextToken();// '*'
   TokenInstance* targs = p.getNextToken();

   // todo: generate an anonymous name
   SynFunc* func = new SynFunc( "", 0, tf->line() );
   if( isEta ) func->setEta(true);
   NameList* list=static_cast<NameList*>(targs->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti= TokenInstance::alloc(tf->line(),tf->chr(), sp.Expr);

   // give the context the occasion to say something about this item
   Expression* expr= new ExprValue( Item( func->handler(), func ), tf->line(),tf->chr() );
   ti->setValue(expr,treestep_deletor);

   // remove this stuff from the stack
   p.simplify( isEta ? 6 : 5,ti);

   // tell the owner we have a new function around.
   ctx->onOpenFunc(func);
   // open a new main state for the function
   ctx->openFunc(func);
   // will check on close if the function is a predicate.
   p.pushState( "InlineFunc", on_close_function , &p );
}


void apply_expr_func( const NonTerminal& r, Parser& p)
{
   //<< T_function << T_Openpar << ListSymbol << T_Closepar << T_EOL
   internal_expr_func( r, p, false );   
}

void apply_expr_funcEta( const NonTerminal& r, Parser& p)
{
   //<< T_function << T_Times << T_Openpar << ListSymbol << T_Closepar << T_EOL
   internal_expr_func( r, p, true );   
}


void apply_return_doubt( const NonTerminal&, Parser& p)
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

void apply_return_eval( const NonTerminal&, Parser& p)
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


void apply_return_break( const NonTerminal&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* texpr= p.getNextToken();//T_return

   StmtReturn* stmt_ret = new StmtReturn( texpr->line(), texpr->chr() );
   stmt_ret->hasBreak(true);
   ctx->addStatement(stmt_ret);
   p.simplify(3);
}

void apply_return_expr( const NonTerminal&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   p.getNextToken();//T_return
   TokenInstance* texpr = p.getNextToken(); // Expr
   
   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   StmtReturn* stmt_ret = new StmtReturn( expr, texpr->line(), texpr->chr() );
   ctx->addStatement(stmt_ret);

   p.simplify(3);
}


void apply_return( const NonTerminal&, Parser& p)
{
   TokenInstance* texpr = p.getNextToken();
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   ctx->addStatement(new StmtReturn(texpr->line(), texpr->chr()) );

   p.simplify(2);
}


void apply_expr_lambda( const NonTerminal&, Parser& p)
{
   // T_OpenGraph
   p.simplify(1);
   p.pushState( "LambdaStart", false );
}

void apply_expr_ep( const NonTerminal&, Parser& p)
{
   // T_CapPar
   TokenInstance* ti = p.getNextToken();
   ExprLit* lit = new ExprLit(ti->line(),ti->chr());

   p.simplify(1);
   static_cast<ParserContext*>(p.context())->openLitContext(lit);
   p.pushState( "EPState", false );
}


void apply_ep_body( const NonTerminal&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   SourceParser& sp = *static_cast<SourceParser*>(&p);

   // << ListExpr << T_Closepar
   TokenInstance* ti = p.getNextToken();
   ExprLit* lit = ctx->closeLitContext();

   ExprEP* ep = new ExprEP( lit->sr().line(), lit->sr().chr()) ;

   List* list = static_cast<List*>(ti->asData());
   for(List::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      Expression* expr = *it;
      ep->append( expr );
   }
   // detach all the expressions.
   list->clear();

   lit->setChild(ep);
   // transform the topmost stack token.
   ti->setValue( lit, treestep_deletor );
   ti->token(sp.Expr);
   p.trim(1);

   // remove the ep start.
   p.popState();
}

static void internal_lambda_params( const NonTerminal&, Parser& p, bool isEta )
{
   // ListSymbol << T_Arrow
   SourceParser& sp = static_cast<SourceParser&>(p);

   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* lsym = sp.getNextToken();
   // always use a real token to forward the line/char pair,
   // as the ListSymbol token may be generated and have 0.
   // that would kill autoexpression generation.
   TokenInstance* tarr = sp.getNextToken();
   
   // and add the function state.
   SynFunc* func = new SynFunc("", 0, tarr->line());
   if( isEta ) func->setEta(true);
   NameList* list = static_cast<NameList*>(lsym->asData());

   for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      func->addParam(*it);
   }

   TokenInstance* ti = TokenInstance::alloc(tarr->line(),tarr->chr(), sp.Expr);
   Expression* expr = new ExprValue( Item(func->handler(), func), tarr->line(),tarr->chr() );
   ti->setValue(expr,treestep_deletor);

   // remove this stuff from the stack
   p.simplify(2,ti);
   // remove the lambdastart state
   p.popState();

   // tell our owner
   ctx->onOpenFunc(func);
   // non-interactive compiler must go on even on error.
   ctx->openFunc(func);
   p.pushState( "InlineFunc", on_close_lambda , &p );
}

void apply_lambda_params( const NonTerminal& r, Parser& p)
{
   internal_lambda_params( r, p, false );
}

void apply_lambda_params_eta( const NonTerminal& r, Parser& p)
{
   internal_lambda_params( r, p, true );
}


void internal_lit_params( const NonTerminal&, Parser& p, bool isEta )
{
   // << T_Openpar << ListSymbol << T_Closepar
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();
   TokenInstance* tparams = p.getNextToken();
   
   // open a fake statement context
   StmtTempLit* tlit = new StmtTempLit();
   tlit->m_forming = new SynTree(ti->line(), ti->chr());

   // but actually we'll be using our lit
   NameList* list=static_cast<NameList*>(tparams->asData());
   ExprLit* lit;
   if( ! list->empty() )
   {
      ExprTree* et = new ExprTree(ti->line(), ti->chr());
      SymbolMap* params = new SymbolMap;

      for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
      {
         params->insert(*it);
      }

      et->setParameters(params);
      if( isEta ) {
         et->setEta( true );
      }

      lit = new ExprLit(et, ti->line(),ti->chr());
   }
   else {
      // {[]  <expr> } is a totally literal expression, without an ExprTree
      if( ! isEta ) {
         lit = new ExprLit(new ExprTree(ti->line(), ti->chr()), ti->line(),ti->chr());
      }
      else {
         lit = new ExprLit(ti->line(),ti->chr());
      }
   }
   
   // (,list,)
   ti = TokenInstance::alloc( ti->line(), ti->chr(), sp.Expr);
   ti->setValue( lit, treestep_deletor );
   p.simplify(3,ti);
   // Use the left "(" as our expression.
   
   // remove the lambdastart state
   p.popState();
   
   ctx->openLitContext(lit);
   ctx->openBlock( tlit, tlit->m_forming );
   p.pushState( "InlineFunc", on_close_lit , &p );
}

void apply_lit_params_eta( const NonTerminal& r, Parser& p)
{
   internal_lit_params( r, p, true );
}

void apply_lit_params( const NonTerminal& r, Parser& p)
{
   internal_lit_params( r, p, false );
}

}

/* end of parser_function.cpp */
