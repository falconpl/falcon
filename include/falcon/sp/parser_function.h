/*
   FALCON - The Falcon Programming Language.
   FILE: parser_function.h

   Parser for Falcon source files -- function declarations handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_FUNCTION_H_
#define _FALCON_SP_PARSER_FUNCTION_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_function( const NonTerminal& r,Parser& p);
void apply_function_eta( const NonTerminal& r,Parser& p);
void apply_static_function( const NonTerminal& r,Parser& p);
void apply_static_function_eta( const NonTerminal& r,Parser& p);
void apply_expr_func( const NonTerminal&, Parser& p);
void apply_expr_funcEta( const NonTerminal&, Parser& p);
void apply_return_doubt( const NonTerminal&, Parser& p);
void apply_return_eval( const NonTerminal&, Parser& p);
void apply_return_break( const NonTerminal&, Parser& p);
void apply_return_expr( const NonTerminal&, Parser& p);
void apply_return( const NonTerminal&, Parser& p);
void apply_expr_lambda( const NonTerminal&, Parser& p);
void apply_expr_ep( const NonTerminal&, Parser& p);
void apply_ep_body( const NonTerminal&, Parser& p);

void apply_lit_params( const NonTerminal&, Parser& p);
void apply_lit_params_eta( const NonTerminal&, Parser& p);
void apply_lambda_params( const NonTerminal&, Parser& p);
void apply_lambda_params_eta( const NonTerminal&, Parser& p);
}

#endif

/* end of parser_function.h */
