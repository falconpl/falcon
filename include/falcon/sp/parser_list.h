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

bool ListExpr_errhand(const NonTerminal&, Parser& p, int);
bool PrintExpr_errhand(const NonTerminal&, Parser& p, int);

void apply_ListExpr_next( const NonTerminal&, Parser& p );
void apply_ListExpr_next_no_comma( const NonTerminal&, Parser& p );
void apply_ListExpr_next2( const NonTerminal&, Parser& p );
void apply_ListExpr_first( const NonTerminal&, Parser& p );
void apply_ListExpr_empty( const NonTerminal&, Parser& p );

void apply_NeListExpr_assign( const NonTerminal&, Parser& p );
void apply_NeListExpr_next( const NonTerminal&, Parser& p );
void apply_NeListExpr_first( const NonTerminal&, Parser& p );

void apply_NeListExpr_ungreed_next( const NonTerminal&, Parser& p );
void apply_NeListExpr_ungreed_first( const NonTerminal&, Parser& p );

//==========================================================
// PairLists
//==========================================================

void apply_ListExprOrPairs_next_pair( const NonTerminal&, Parser& p );
void apply_ListExprOrPairs_next( const NonTerminal&, Parser& p );
void apply_ListExprOrPairs_first_pair( const NonTerminal&, Parser& p );
void apply_ListExprOrPairs_first( const NonTerminal&, Parser& p );
void apply_ListExprOrPairs_empty( const NonTerminal&, Parser& p );

//==========================================================
// SeqPairList
//==========================================================

void apply_SeqExprOrPairs_next_pair_cm( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_next_pair( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_next( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_next_cm( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_first_pair( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_first( const NonTerminal&, Parser& p );
void apply_SeqExprOrPairs_empty( const NonTerminal&, Parser& p );

void apply_ListSymbol_first( const NonTerminal&,Parser& p);
void apply_ListSymbol_next( const NonTerminal&,Parser& p);
void apply_ListSymbol_next2( const NonTerminal&,Parser& p);
void apply_ListSymbol_empty( const NonTerminal&,Parser& p);

void apply_NeListSymbol_first( const NonTerminal&, Parser& p);
void apply_NeListSymbol_next( const NonTerminal&, Parser& p);

}

#endif

/* end of parser_list.h */
