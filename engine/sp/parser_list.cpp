/*
   FALCON - The Falcon Programming Language.
   FILE: parser_list.cpp

   Parser for Falcon source files -- index accessor handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:04:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_list.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/trace.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_index.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/parser/parser.h>

#include <falcon/expression.h>
#include "private_types.h"

#include <falcon/sp/parser_list.h>

namespace Falcon {

using namespace Parsing;

bool ListExpr_errhand(const NonTerminal&, Parser& p, int)
{
   TRACE2( "ListExpr_errhand -- removing %d tokens", p.tokenCount() );
   TokenInstance* t0 = p.getNextToken();
   TokenInstance* t1 = p.getLastToken();

   p.addError( e_syn_list_decl, p.currentSource(), t1->line(), t1->chr(), t0->line() );
   p.setErrorMode(&p.T_EOL);
   return true;
}

bool PrintExpr_errhand(const NonTerminal&, Parser& p, int)
{
   TRACE2( "PrintExpr_errhand -- removing %d tokens", p.tokenCount() );
   TokenInstance* t0 = p.getNextToken();
   TokenInstance* t1 = p.getLastToken();

   p.addError( e_syn_self_print, p.currentSource(), t1->line(), t1->chr(), t0->line() );
   p.setErrorMode(&p.T_EOL);
   return true;
}

void apply_ListExpr_next( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   List* list = static_cast<List*>(tlist->asData());
   list->push_back(expr);
   tlist->token(sp.ListExpr);
   p.trimFromBase(1,2);
}

void apply_ListExpr_next_no_comma( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   List* list = static_cast<List*>(tlist->asData());
   list->push_back(expr);
   tlist->token(sp.ListExpr);
   p.trimFromBase(1,1);
}


void apply_ListExpr_next2( const NonTerminal&, Parser& p )
{
   // << ListExpr << T_EOL
   // remove the trailing eol
   p.trim(1);
}

void apply_ListExpr_first( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   List* list = new List;
   list->push_back(expr);

   // Change the expression in a list
   texpr->token(sp.ListExpr);
   texpr->setValue( list, list_deletor );
}

void apply_ListExpr_empty( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = TokenInstance::alloc(0, 0, sp.ListExpr );

   List* list = new List;
   ti_list->setValue( list, list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}

void apply_NeListExpr_assign( const NonTerminal&, Parser& )
{
   // does nothing
}

void apply_NeListExpr_next( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   List* list = static_cast<List*>(tlist->asData());
   list->push_back(expr);
   tlist->token( sp.NeListExpr );
   p.trimFromBase(1,2);
}

void apply_NeListExpr_first( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());;

   List* list = new List;
   list->push_back(expr);

   // Change the expression in a list
   texpr->token( sp.NeListExpr );
   texpr->setValue( list, list_deletor );
}


void apply_NeListExpr_ungreed_next( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   List* list = static_cast<List*>(tlist->asData());
   list->push_back(expr);
   // we must create a new token as the expression is ungreedy and we have more things in the stack.
   tlist->token( sp.NeListExpr_ungreed);
   p.trimFromBase(1,2);
}


void apply_NeListExpr_ungreed_first( const NonTerminal&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   List* list = new List;
   list->push_back(expr);

   // Change the expression in a list
   texpr->token( sp.NeListExpr_ungreed );
   texpr->setValue( list, list_deletor );
}


void apply_ListSymbol_first( const NonTerminal&,Parser& p)
{
   // << (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   // change the symbol in list
   NameList* list = new NameList;
   list->push_back(*tname->asString());
   tname->token(sp.ListSymbol);
   tname->setValue( list, name_list_deletor );
}


void apply_ListSymbol_next( const NonTerminal&,Parser& p)
{
   // << (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->asData());
   list->push_back(*tname->asString());

   tlist->token(sp.ListSymbol);
   p.trimFromBase(1,2);
}

void apply_ListSymbol_next2( const NonTerminal&,Parser& p)
{
   // << ListSymbol << T_EOL
   // Just remove the eol
   p.trim(1);
}


void apply_ListSymbol_empty( const NonTerminal&,Parser& p)
{
   // << (r_ListSymbol_empty << "ListSymbol_empty" << apply_ListSymbol_empty )

   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = TokenInstance::alloc(0, 0, sp.ListSymbol);

   NameList* list=new NameList;
   ti_list->setValue( list, name_list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}


void apply_NeListSymbol_first( const NonTerminal&, Parser& p)
{
   // << (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   NameList* list = new NameList;
   list->push_back(*tname->asString());
   tname->token( sp.NeListSymbol);
   tname->setValue( list, name_list_deletor );
}


void apply_NeListSymbol_next( const NonTerminal&, Parser& p)
{
   // << (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->asData());
   list->push_back(*tname->asString());
   tlist->token(sp.NeListSymbol);

   p.trimFromBase(1,2);
}

}

/* end of parser_list.cpp */
