/*
   FALCON - The Falcon Programming Language.
   FILE: parser_arraydecl.h

   Parser for Falcon source files -- handler for array and dict decl
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_ARRAYDECL_H_
#define _FALCON_SP_PARSER_ARRAYDECL_H_

#include <falcon/setup.h>

namespace Falcon {

class BinaryExpression;

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_array_decl( const Rule&, Parser& p );
void apply_expr_empty_dict( const Rule&, Parser& p );

}

#endif

/* end of parser_arraydecl.h */
