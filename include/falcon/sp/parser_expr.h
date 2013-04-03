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

bool expr_errhand(const NonTerminal&, Parser& p);

void apply_expr_equal( const Rule& r, Parser& p );
void apply_expr_diff( const Rule& r, Parser& p );
void apply_expr_less( const Rule& r, Parser& p );
void apply_expr_greater( const Rule& r, Parser& p );
void apply_expr_le( const Rule& r, Parser& p );
void apply_expr_ge( const Rule& r, Parser& p );
void apply_expr_eeq( const Rule& r, Parser& p );
void apply_expr_plus( const Rule& r, Parser& p );
void apply_expr_in( const Rule& r, Parser& p );
void apply_expr_notin( const Rule& r, Parser& p );

void apply_expr_preinc(const Rule&, Parser& p );
void apply_expr_postinc(const Rule&, Parser& p );
void apply_expr_predec(const Rule&, Parser& p );
void apply_expr_postdec(const Rule&, Parser& p );

void apply_expr_minus( const Rule& r, Parser& p );

void apply_expr_times( const Rule& r, Parser& p );
void apply_expr_div( const Rule& r, Parser& p );
void apply_expr_mod( const Rule& r, Parser& p );
void apply_expr_pow( const Rule& r, Parser& p );
void apply_expr_shr( const Rule&, Parser& p );
void apply_expr_shl( const Rule&, Parser& p );

void apply_expr_and( const Rule&, Parser& p );
void apply_expr_or( const Rule&, Parser& p );

void apply_expr_band( const Rule&, Parser& p );
void apply_expr_bor( const Rule&, Parser& p );
void apply_expr_bxor( const Rule&, Parser& p );

void apply_expr_invoke( const Rule&, Parser& p );
void apply_expr_compose( const Rule&, Parser& p );
void apply_expr_funcpower( const Rule&, Parser& p );

void apply_expr_neg( const Rule&, Parser& p );
void apply_expr_not( const Rule&, Parser& p );
void apply_expr_bnot( const Rule&, Parser& p );
void apply_expr_oob( const Rule&, Parser& p );
void apply_expr_deoob( const Rule&, Parser& p );
void apply_expr_xoob( const Rule&, Parser& p );
void apply_expr_isoob( const Rule&, Parser& p );
void apply_expr_str_ipol( const Rule&, Parser& p );
void apply_expr_eval( const Rule&, Parser& p );
void apply_expr_unquote( const Rule&, Parser& p );

void apply_expr_evalret( const Rule&r, Parser& p );
void apply_expr_evalret_exec( const Rule&r, Parser& p );
void apply_expr_evalret_doubt( const Rule&r, Parser& p );


void apply_expr_auto_add( const Rule&, Parser& p );
void apply_expr_auto_sub( const Rule&, Parser& p );
void apply_expr_auto_times( const Rule&, Parser& p );
void apply_expr_auto_div( const Rule&, Parser& p );
void apply_expr_auto_mod( const Rule&, Parser& p );
void apply_expr_auto_pow( const Rule&, Parser& p );
void apply_expr_auto_shr( const Rule&, Parser& p );
void apply_expr_auto_shl( const Rule&, Parser& p );


void apply_expr_pars( const Rule&, Parser& p );
void apply_expr_dot( const Rule&, Parser& p );
void apply_expr_provides( const Rule&, Parser& p );

}

#endif	/* _FALCON_SP_PARSER_EXPR_H_ */

/* end of parser_expr.h */
