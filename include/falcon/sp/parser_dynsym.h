/*
   FALCON - The Falcon Programming Language.
   FILE: parser_dynsym.h

   Falcon source parser -- Deletors used in parsing expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 11 Jan 2012 11:15:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SP_PARSER_DYNSYM_H_
#define _FALCON_SP_PARSER_DELETOR_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_expr_symname( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_dynsym.h */
