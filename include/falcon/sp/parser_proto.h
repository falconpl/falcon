/*
   FALCON - The Falcon Programming Language.
   FILE: parser_proto.h

   Parser for Falcon source files -- prototype declarations handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_PROTO_H_
#define _FALCON_SP_PARSER_PROTO_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_proto( const NonTerminal&, Parser& p);
void apply_proto_prop( const NonTerminal&, Parser& p);

}

#endif

/* end of parser_proto.h */
