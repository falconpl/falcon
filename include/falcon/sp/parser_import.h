/*
   FALCON - The Falcon Programming Language.
   FILE: parser_import.h

   Parser for Falcon source files -- import/from directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 16:41:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_IMPORT_H_
#define _FALCON_SP_PARSER_IMPORT_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool import_errhand(const NonTerminal&, Parser& p, int);

void apply_import( const NonTerminal&, Parser& p );

void apply_import_from_string_as( const NonTerminal&, Parser& p );
void apply_import_from_string_in( const NonTerminal&, Parser& p );
void apply_import_star_from_string_in( const NonTerminal&, Parser& p );
void apply_import_star_from_string( const NonTerminal&, Parser& p );
void apply_import_string( const NonTerminal&, Parser& p );

void apply_import_from_modspec_as( const NonTerminal&, Parser& p );
void apply_import_from_modspec_in( const NonTerminal&, Parser& p );
void apply_import_star_from_modspec_in( const NonTerminal&, Parser& p );
void apply_import_star_from_modspec( const NonTerminal&, Parser& p );
void apply_import_from_modspec( const NonTerminal&, Parser& p );

void apply_import_syms( const NonTerminal&, Parser& p );

bool importspec_errhand(const NonTerminal&, Parser& p, int);

void apply_ImportSpec_next( const NonTerminal&, Parser& p );
void apply_ImportSpec_attach_last( const NonTerminal&, Parser& p );
void apply_ImportSpec_attach_next( const NonTerminal&, Parser& p );
void apply_ImportSpec_first( const NonTerminal&, Parser& p );
void apply_ImportSpec_empty( const NonTerminal&, Parser& p );

void apply_nsspec_last( const NonTerminal&, Parser& p );
void apply_nsspec_next( const NonTerminal&, Parser& p );
void apply_nsspec_first( const NonTerminal&, Parser& p );
}

#endif

/* end of parser_import.h */
