/*
   FALCON - The Falcon Programming Language.
   FILE: parser_summon.h

   Falcon source parser -- summon expression handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 17 Apr 2013 17:20:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_SUMMON_H_
#define	_FALCON_SP_PARSER_SUMMON_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing {
class Rule;
class Parser;
}

void apply_expr_summon( const Parsing::NonTerminal&, Parsing::Parser& p );
void apply_expr_opt_summon( const Parsing::NonTerminal&, Parsing::Parser& p );

}

#endif

/* end of parser_summon.h */
