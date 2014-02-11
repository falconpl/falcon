/*
   FALCON - The Falcon Programming Language.
   FILE: parser_autoexpr.h

   Parser for Falcon source files -- autoexpression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_AUTOEXPR_H_
#define _FALCON_SP_PARSER_AUTOEXPR_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_line_expr( const NonTerminal&, Parser& p );
void apply_autoexpr_list( const NonTerminal&, Parser& p );
void apply_stmt_assign_list( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_autoexpr.h */
