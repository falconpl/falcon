/*
   FALCON - The Falcon Programming Language.
   FILE: parser_try.h

   Parser for Falcon source files -- try statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:32:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_TRY_H_
#define _FALCON_SP_PARSER_TRY_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool try_errand(const NonTerminal&, Parser& p);
bool catch_errhand(const NonTerminal&, Parser& p);
bool finally_errhand(const NonTerminal&, Parser& p);
bool rasie_errhand(const NonTerminal&, Parser& p);

void apply_try( const Rule&, Parser& p );
void apply_catch( const Rule&, Parser& p );
void apply_finally( const Rule&, Parser& p );
void apply_raise( const Rule&, Parser& p );


void apply_catch_all( const Rule&, Parser& p );
void apply_catch_in_var( const Rule&, Parser& p );
void apply_catch_number( const Rule&, Parser& p );
void apply_catch_number_in_var( const Rule&, Parser& p );
void apply_catch_thing( const Rule&, Parser& p );
void apply_catch_thing_in_var( const Rule&, Parser& p );

}

#endif

/* end of parser_try.h */
