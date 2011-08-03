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

bool import_errhand(const NonTerminal&, Parser& p);

void apply_import( const Rule&, Parser& p );

void apply_import_from_string_as( const Rule&, Parser& p );
void apply_import_from_string_in( const Rule&, Parser& p );
void apply_import_string( const Rule&, Parser& p );

void apply_import_from_modspec_as( const Rule&, Parser& p );
void apply_import_from_modspec_in( const Rule&, Parser& p );
void apply_import_from_modspec( const Rule&, Parser& p );

void apply_import_syms( const Rule&, Parser& p );

}

#endif

/* end of parser_import.h */
