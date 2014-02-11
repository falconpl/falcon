/*
   FALCON - The Falcon Programming Language.
   FILE: parser_attribute.h

   Parser for Falcon source files -- attribute declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Jan 2013 18:17:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_ATTRIBUTE_H_
#define _FALCON_SP_PARSER_ATTRIBUTE_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool errhand_attribute(const NonTerminal&, Parser& p, int);

void apply_attribute( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_attribute.h */
