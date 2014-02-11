/*
   FALCON - The Falcon Programming Language.
   FILE: parser_export.h

   Parser for Falcon source files -- export dirrective
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Jan 2013 16:29:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_GLOBAL_H_
#define _FALCON_SP_PARSER_GLOBAL_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool global_errhand(const NonTerminal&, Parser& p);
void apply_global( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_global.h */
