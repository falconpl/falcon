/*
   FALCON - The Falcon Programming Language.
   FILE: parser_export.h

   Parser for Falcon source files -- export dirrective
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 16:41:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#ifndef _FALCON_SP_PARSER_EXPORT_H_
#define _FALCON_SP_PARSER_EXPORT_H_

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool export_errhand(const NonTerminal&, Parser& p, int);
void apply_export_rule( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_export.h */
