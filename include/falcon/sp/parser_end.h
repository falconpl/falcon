/*
   FALCON - The Falcon Programming Language.
   FILE: parser_end.h

   Parser for Falcon source files -- end statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 19:10:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_SP_PARSER_END_H_
#define _FALCON_SP_PARSER_END_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_end( const NonTerminal&, Parser& p );
void apply_end_small( const NonTerminal&, Parser& p );
void apply_end_rich( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_end.h */
