/*
   FALCON - The Falcon Programming Language.
   FILE: parser_list.h

   Parser for Falcon source files -- index accessor handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:04:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_LIST_H_
#define _FALCON_SP_PARSER_LIST_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool ListExpr_errhand(const NonTerminal&, Parser& p);
bool PrintExpr_errhand(const NonTerminal&, Parser& p);

void apply_ListExpr_next( const Rule&, Parser& p );
void apply_ListExpr_next_no_comma( const Rule&, Parser& p );
void apply_ListExpr_next2( const Rule&, Parser& p );
void apply_ListExpr_first( const Rule&, Parser& p );
void apply_ListExpr_empty( const Rule&, Parser& p );

void apply_NeListExpr_next( const Rule&, Parser& p );
void apply_NeListExpr_first( const Rule&, Parser& p );

void apply_NeListExpr_ungreed_next( const Rule&, Parser& p );
void apply_NeListExpr_ungreed_first( const Rule&, Parser& p );

//==========================================================
// PairLists
//==========================================================

void apply_ListExprOrPairs_next_pair( const Rule&, Parser& p );
void apply_ListExprOrPairs_next( const Rule&, Parser& p );
void apply_ListExprOrPairs_first_pair( const Rule&, Parser& p );
void apply_ListExprOrPairs_first( const Rule&, Parser& p );
void apply_ListExprOrPairs_empty( const Rule&, Parser& p );

//==========================================================
// SeqPairList
//==========================================================

void apply_SeqExprOrPairs_next_pair_cm( const Rule&, Parser& p );
void apply_SeqExprOrPairs_next_pair( const Rule&, Parser& p );
void apply_SeqExprOrPairs_next( const Rule&, Parser& p );
void apply_SeqExprOrPairs_next_cm( const Rule&, Parser& p );
void apply_SeqExprOrPairs_first_pair( const Rule&, Parser& p );
void apply_SeqExprOrPairs_first( const Rule&, Parser& p );
void apply_SeqExprOrPairs_empty( const Rule&, Parser& p );

void apply_ListSymbol_first(const Rule&,Parser& p);
void apply_ListSymbol_next(const Rule&,Parser& p);
void apply_ListSymbol_next2(const Rule&,Parser& p);
void apply_ListSymbol_empty(const Rule&,Parser& p);

void apply_NeListSymbol_first(const Rule&, Parser& p);
void apply_NeListSymbol_next(const Rule&, Parser& p);

}

#endif

/* end of parser_list.h */
