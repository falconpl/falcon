/*
   FALCON - The Falcon Programming Language.
   FILE: parser_if.h

   Parser for Falcon source files -- if statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_IF_H_
#define _FALCON_SP_PARSER_IF_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_if_short( const Rule&, Parser& p );
void apply_if( const Rule&, Parser& p );
void apply_elif( const Rule&, Parser& p );
void apply_else( const Rule&, Parser& p );

}

#endif

/* end of parser_if.h */
