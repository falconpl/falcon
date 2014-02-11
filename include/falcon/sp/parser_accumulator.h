/*
   FALCON - The Falcon Programming Language.
   FILE: parser_accumulator.h

   Parser for Falcon source files -- Accumulator expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 04 Apr 2013 22:22:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_ACCUMULATOR_H_
#define _FALCON_SP_PARSER_ACCUMULATOR_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_accumulator( const NonTerminal& r,Parser& p);

void apply_accumulator_complete( const NonTerminal& r,Parser& p);
void apply_accumulator_w_target( const NonTerminal& r,Parser& p);
void apply_accumulator_w_filter( const NonTerminal& r,Parser& p);
void apply_accumulator_simple( const NonTerminal& r,Parser& p);

}

#endif

/* end of parser_accumulator.h */
