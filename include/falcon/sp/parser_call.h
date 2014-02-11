/*
   FALCON - The Falcon Programming Language.
   FILE: parser_call.h

   Falcon source parser -- function call handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:51:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_CALL_H_
#define	_FALCON_SP_PARSER_CALL_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing {
class Rule;
class Parser;
}

void apply_expr_call( const Parsing::NonTerminal&, Parsing::Parser& p );

}

#endif

/* end of parser_call.h */
