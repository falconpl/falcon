/*
   FALCON - The Falcon Programming Language.
   FILE: parser_for.h

   Parser for Falcon source files -- for statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Aug 2011 16:56:41 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_FOR_H_
#define _FALCON_SP_PARSER_FOR_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool for_errhand(const NonTerminal&, Parser& p, int);

void apply_for_to_step( const NonTerminal&, Parser& p );
void apply_for_to_step_short( const NonTerminal&, Parser& p );
void apply_for_to( const NonTerminal&, Parser& p );
void apply_for_to_short( const NonTerminal&, Parser& p );
void apply_for_in( const NonTerminal&, Parser& p );
void apply_for_in_short( const NonTerminal&, Parser& p );
      
void apply_forfirst( const NonTerminal&, Parser& p );
void apply_forfirst_short( const NonTerminal&, Parser& p );
void apply_formiddle( const NonTerminal&, Parser& p );
void apply_formiddle_short( const NonTerminal&, Parser& p );
void apply_forlast( const NonTerminal&, Parser& p );
void apply_forlast_short( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_for.h */
