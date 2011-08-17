/*
   FALCON - The Falcon Programming Language.
   FILE: parser_reference.h

   Parser for Falcon source files -- reference
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 11:41:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_REFERENCE_H_
#define _FALCON_SP_PARSER_REFERENCE_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_ref( const Rule&, Parser& p );

}

#endif

/* end of parser_reference.h */
