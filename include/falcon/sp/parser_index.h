/*
   FALCON - The Falcon Programming Language.
   FILE: parser_index.h

   Falcon source parser -- expression parsing handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_INDEX_H_
#define _FALCON_SP_PARSER_INDEX_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

void apply_expr_index( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_star_index( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_range_index3( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_range_index3open( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_range_index2( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_range_index1( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_range_index0( const Parsing::NonTerminal&, Parsing::Parser& p );


}

#endif

/* end of parser_index.h */
