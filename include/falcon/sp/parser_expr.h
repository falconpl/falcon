/*
   FALCON - The Falcon Programming Language.
   FILE: parser_expr.h

   Falcon source parser -- expression parsing handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_EXPR_H_
#define _FALCON_SP_PARSER_EXPR_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

class BinaryExpression;

using namespace Parsing;

bool expr_errhand(const NonTerminal&, Parser& p, int);

void apply_expr_equal( const NonTerminal& r, Parser& p );
void apply_expr_diff( const NonTerminal& r, Parser& p );
void apply_expr_less( const NonTerminal& r, Parser& p );
void apply_expr_greater( const NonTerminal& r, Parser& p );
void apply_expr_le( const NonTerminal& r, Parser& p );
void apply_expr_ge( const NonTerminal& r, Parser& p );
void apply_expr_eeq( const NonTerminal& r, Parser& p );
void apply_expr_plus( const NonTerminal& r, Parser& p );
void apply_expr_in( const NonTerminal& r, Parser& p );
void apply_expr_notin( const NonTerminal& r, Parser& p );

void apply_expr_preinc( const NonTerminal&, Parser& p );
void apply_expr_postinc( const NonTerminal&, Parser& p );
void apply_expr_predec( const NonTerminal&, Parser& p );
void apply_expr_postdec( const NonTerminal&, Parser& p );

void apply_expr_minus( const NonTerminal& r, Parser& p );

void apply_expr_times( const NonTerminal& r, Parser& p );
void apply_expr_div( const NonTerminal& r, Parser& p );
void apply_expr_mod( const NonTerminal& r, Parser& p );
void apply_expr_pow( const NonTerminal& r, Parser& p );
void apply_expr_shr( const NonTerminal&, Parser& p );
void apply_expr_shl( const NonTerminal&, Parser& p );

void apply_expr_and( const NonTerminal&, Parser& p );
void apply_expr_or( const NonTerminal&, Parser& p );

void apply_expr_band( const NonTerminal&, Parser& p );
void apply_expr_bor( const NonTerminal&, Parser& p );
void apply_expr_bxor( const NonTerminal&, Parser& p );

void apply_expr_invoke( const NonTerminal&, Parser& p );
void apply_expr_compose( const NonTerminal&, Parser& p );
void apply_expr_funcpower( const NonTerminal&, Parser& p );

void apply_expr_neg( const NonTerminal&, Parser& p );
void apply_expr_not( const NonTerminal&, Parser& p );
void apply_expr_bnot( const NonTerminal&, Parser& p );
void apply_expr_oob( const NonTerminal&, Parser& p );
void apply_expr_deoob( const NonTerminal&, Parser& p );
void apply_expr_xoob( const NonTerminal&, Parser& p );
void apply_expr_isoob( const NonTerminal&, Parser& p );
void apply_expr_str_ipol( const NonTerminal&, Parser& p );
void apply_expr_eval( const NonTerminal&, Parser& p );
void apply_expr_unquote( const NonTerminal&, Parser& p );

void apply_expr_evalret( const NonTerminal&r, Parser& p );
void apply_expr_evalret_exec( const NonTerminal&r, Parser& p );
void apply_expr_evalret_doubt( const NonTerminal&r, Parser& p );


void apply_expr_auto_add( const NonTerminal&, Parser& p );
void apply_expr_auto_sub( const NonTerminal&, Parser& p );
void apply_expr_auto_times( const NonTerminal&, Parser& p );
void apply_expr_auto_div( const NonTerminal&, Parser& p );
void apply_expr_auto_mod( const NonTerminal&, Parser& p );
void apply_expr_auto_pow( const NonTerminal&, Parser& p );
void apply_expr_auto_shr( const NonTerminal&, Parser& p );
void apply_expr_auto_shl( const NonTerminal&, Parser& p );


void apply_expr_pars( const NonTerminal&, Parser& p );
void apply_expr_dot( const NonTerminal&, Parser& p );
void apply_expr_provides( const NonTerminal&, Parser& p );

void apply_expr_named( const NonTerminal&, Parser& p );
}

#endif	/* _FALCON_SP_PARSER_EXPR_H_ */

/* end of parser_expr.h */
