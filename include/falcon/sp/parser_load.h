/*
   FALCON - The Falcon Programming Language.
   FILE: parser_load.h

   Parser for Falcon source files -- index accessor handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 14:31:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_LOAD_H_
#define _FALCON_SP_PARSER_LOAD_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool load_errhand(const NonTerminal&, Parser& p, int);

void apply_load_string( const NonTerminal&, Parser& p );
void apply_load_mod_spec( const NonTerminal&, Parser& p );

bool load_modspec_errhand(const NonTerminal&, Parser& p, int);

void apply_modspec_next( const NonTerminal&, Parser& p );
void apply_modspec_first( const NonTerminal&, Parser& p );
void apply_modspec_first_self( const NonTerminal&, Parser& p );
void apply_modspec_first_dot( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_load.h */
