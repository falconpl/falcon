/*
   FALCON - The Falcon Programming Language.
   FILE: namespace_import.h

   Parser for Falcon source files -- import/from directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 04 Aug 2011 02:22:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_NAMESPACE_H_
#define _FALCON_SP_PARSER_NAMESPACE_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool namespace_errhand(const NonTerminal&, Parser& p, int);

void apply_namespace( const NonTerminal&, Parser& p );
}

#endif

/* end of parser_namespace.h */
