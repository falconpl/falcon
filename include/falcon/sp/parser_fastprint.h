/*
   FALCON - The Falcon Programming Language.
   FILE: parser_fastprint.h

   Parser for Falcon source files -- fastprint statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Aug 2011 12:41:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_FASTPRINT_H_
#define _FALCON_SP_PARSER_FASTPRINT_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_fastprint( const NonTerminal&, Parser& p );
void apply_fastprint_nl( const NonTerminal&, Parser& p );
void apply_fastprint_alone( const NonTerminal&, Parser& p );
void apply_fastprint_nl_alone( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_fastprint.h */
