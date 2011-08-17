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

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/expression.h>
#include "private_types.h"

#include <falcon/sp/parser_list.h>

namespace Falcon {

using namespace Parsing;

bool ListExpr_errhand(const NonTerminal&, Parser& p)
{
   TRACE2( "ListExpr_errhand -- removing %d tokens", p.tokenCount() );
   TokenInstance* t0 = p.getNextToken();
   TokenInstance* t1 = p.getLastToken();

   p.addError( e_syn_arraydecl, p.currentSource(), t1->line(), t1->chr(), t0->line() );
   p.trimFromCurrentToken();
   return true;
}

void apply_ListExpr_next( const Rule&, Parser& p )
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

void apply_ListExpr_first( const Rule&, Parser& p )
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

void apply_NeListExpr_next( const Rule&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   List* list = static_cast<List*>(tlist->detachValue());
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.NeListExpr );
   ti_list->setValue( list, list_deletor );
   p.simplify(3,ti_list);

}

void apply_NeListExpr_first( const Rule&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());;

   List* list = new List;
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(texpr->line(), texpr->chr(), sp.NeListExpr );
   ti_list->setValue( list, list_deletor );

   // Change the expression in a list
   p.simplify( 1, ti_list );
}


void apply_ListExpr_empty( const Rule&, Parser& p )
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


void apply_NeListExpr_ungreed_next( const Rule&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   List* list = static_cast<List*>(tlist->detachValue());
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.NeListExpr_ungreed );
   ti_list->setValue( list, list_deletor );
   p.simplify(3,ti_list);

}


void apply_NeListExpr_ungreed_first( const Rule&, Parser& p )
{
   // << (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << Expr )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* texpr = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());;

   List* list = new List;
   list->push_back(expr);

   TokenInstance* ti_list = new TokenInstance(texpr->line(), texpr->chr(), sp.NeListExpr_ungreed );
   ti_list->setValue( list, list_deletor );

   // Change the expression in a list
   p.simplify( 1, ti_list );
}


void apply_ListSymbol_first(const Rule&,Parser& p)
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


void apply_ListSymbol_next(const Rule&,Parser& p)
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


void apply_ListSymbol_empty(const Rule&,Parser& p)
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


void apply_NeListSymbol_first(const Rule&, Parser& p)
{
   // << (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   TokenInstance* ti_list = new TokenInstance(tname->line(), tname->chr(), sp.NeListSymbol);

   NameList* list = new NameList;
   list->push_back(*tname->asString());
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 1, ti_list );
}


void apply_NeListSymbol_next(const Rule&, Parser& p)
{
   // << (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->detachValue());
   list->push_back(*tname->asString());

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.NeListSymbol );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );
}

}

/* end of parser_list.cpp */
