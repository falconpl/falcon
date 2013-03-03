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

bool load_errhand(const NonTerminal&, Parser& p);

void apply_load_string( const Rule&, Parser& p );
void apply_load_mod_spec( const Rule&, Parser& p );

bool load_modspec_errhand(const NonTerminal&, Parser& p);

void apply_modspec_next( const Rule&, Parser& p );
void apply_modspec_first( const Rule&, Parser& p );
void apply_modspec_first_self( const Rule&, Parser& p );
void apply_modspec_first_dot( const Rule&, Parser& p );

}

#endif

/* end of parser_load.h */
