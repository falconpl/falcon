/*
   FALCON - The Falcon Programming Language.
   FILE: parser_while.h

   Parser for Falcon source files -- while statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_WHILE_H_
#define _FALCON_SP_PARSER_WHILE_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool while_errhand(const NonTerminal&, Parser& p, int i);

void apply_while_short( const NonTerminal&, Parser& p );
void apply_while( const NonTerminal&, Parser& p );
void apply_continue( const NonTerminal&, Parser& p );
void apply_break( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_while.h */
