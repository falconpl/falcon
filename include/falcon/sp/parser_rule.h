/*
   FALCON - The Falcon Programming Language.
   FILE: parser_rule.h

   Parser for Falcon source files -- rule declaration handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_RULE_H_
#define _FALCON_SP_PARSER_RULE_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_rule( const NonTerminal&, Parser& p );
void apply_rule_branch( const NonTerminal&, Parser& p );
void apply_cut_expr( const NonTerminal&, Parser& p );
void apply_cut( const NonTerminal&, Parser& p );
void apply_doubt( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_rule.h */
