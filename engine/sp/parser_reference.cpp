/*
   FALCON - The Falcon Programming Language.
   FILE: parser_reference.cpp

   Parser for Falcon source files -- references
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 11:41:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_reference.cpp"

#include <falcon/setup.h>

#include <falcon/expression.h>
#include <falcon/exprref.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_reference.h>

#include "private_types.h"
#include "falcon/symbol.h"


namespace Falcon {

using namespace Parsing;

void apply_expr_ref( const Rule&, Parser& p )
{
   // << T_Dollar << T_Name
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   // get the tokens
   p.getNextToken();
   TokenInstance* tref = p.getNextToken();
   
   // Create the symbol and the reference
   Symbol* sym = ctx->addVariable(*tref->asString());
   ExprSymbol* refexpr = static_cast<ExprSymbol*>(sym->makeExpression());
   Expression* reference = new ExprRef( refexpr );
   
   // update the result token
   TokenInstance* ti = new TokenInstance( tref->line(), tref->chr(), sp.Expr );
   ti->setValue( reference, expr_deletor );   
   p.simplify(2, ti); 
}

}

/* end of parser_reference.cpp */
