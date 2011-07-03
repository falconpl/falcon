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

//==========================================================
// PairLists
//==========================================================

void apply_ListExprOrPairs_next_pair( const Rule&, Parser& p )
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


void apply_ListExprOrPairs_next( const Rule&, Parser& p )
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


void apply_ListExprOrPairs_first_pair( const Rule&, Parser& p )
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


void apply_ListExprOrPairs_first( const Rule&, Parser& p )
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


void apply_ListExprOrPairs_empty( const Rule&, Parser& p )
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

void apply_SeqExprOrPairs_next_pair_cm( const Rule&, Parser& p )
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


void apply_SeqExprOrPairs_next_pair( const Rule&, Parser& p )
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


void apply_SeqExprOrPairs_next( const Rule&, Parser& p )
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


void apply_SeqExprOrPairs_next_cm( const Rule&, Parser& p )
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
void apply_SeqExprOrPairs_first_pair( const Rule&, Parser& p )
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
void apply_SeqExprOrPairs_first( const Rule&, Parser& p )
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
void apply_SeqExprOrPairs_empty( const Rule&, Parser& p )
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
