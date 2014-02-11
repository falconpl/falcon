/*
   FALCON - The Falcon Programming Language.
   FILE: parser_loop.h

   Parser for Falcon source files -- while statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Feb 2013 12:43:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_LOOP_H_
#define _FALCON_SP_PARSER_LOOP_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool loop_errhand(const NonTerminal&, Parser& p, int);

void apply_loop_short( const NonTerminal&, Parser& p );
void apply_loop( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_loop.h */
