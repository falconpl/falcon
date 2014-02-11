/*
   FALCON - The Falcon Programming Language.
   FILE: parser_ternaryif.h

   Parser for Falcon source files --  ternary if expression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 10 Jan 2012 23:00:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_TERNARYIF_H_
#define _FALCON_SP_PARSER_TERNARYIF_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_ternary_if( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_ternaryif.h */
