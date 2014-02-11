/*
   FALCON - The Falcon Programming Language.
   FILE: parser_class.h

   Parser for Falcon source files -- class statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 19:10:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_SP_PARSER_CLASS_H_
#define _FALCON_SP_PARSER_CLASS_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

bool classdecl_errhand(const NonTerminal&, Parser& p, int);

void apply_class_statement( const NonTerminal&, Parser& p );
void apply_object_statement( const NonTerminal&, Parser& p );

void apply_pdecl_expr( const NonTerminal&, Parser& p );
void apply_static_pdecl_expr( const NonTerminal&, Parser& p );
void apply_init_expr( const NonTerminal&, Parser& p );

void apply_FromClause_next( const NonTerminal&, Parser& p );
void apply_FromClause_first( const NonTerminal&, Parser& p );

void apply_FromClause_entry_with_expr( const NonTerminal&, Parser& p );
void apply_FromClause_entry( const NonTerminal&, Parser& p );


void apply_expr_class( const NonTerminal&, Parser& p );
void apply_class_from( const NonTerminal&, Parser& p );
void apply_class( const NonTerminal&, Parser& p );
void apply_class_p_from( const NonTerminal&, Parser& p );
void apply_class_p( const NonTerminal&, Parser& p );

}

#endif

/* end of parser_class.h */
