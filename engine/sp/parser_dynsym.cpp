/*
   FALCON - The Falcon Programming Language.
   FILE: parser_dynsym.cpp

   Parser for Falcon source files -- dynamic symbol
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 11 Jan 2012 11:15:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_dynsym.cpp"

#include <falcon/setup.h>
#include <falcon/symbol.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_dynsym.h>

#include <falcon/engine.h>
#include <falcon/psteps/exprvalue.h>

#include "private_types.h"
#include "falcon/psteps/exprsym.h"


namespace Falcon {

using namespace Parsing;


void apply_expr_amper( const Rule&, Parser& p )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls = Engine::instance()->symbolClass();
   
   // << T_Amper << T_Name
   SourceParser& sp = static_cast<SourceParser&>(p);

   // get the tokens
   p.getNextToken();
   TokenInstance* tref = p.getNextToken();
   
   // Create the symbol and the reference
   
   // this creates a dynsymbol.
   Symbol* nsym = new Symbol(*tref->asString(), tref->line());
   
   // TODO: Use the engine to cache dynsymbols
   // TODO: Use garbage collector
   FALCON_GC_STORE(coll, cls, nsym );
   ExprSymbol* esyn = new ExprSymbol( tref->line(), tref->chr() );
   esyn->safeGuard( nsym );
   
   // update the result token
   TokenInstance* ti = new TokenInstance( tref->line(), tref->chr(), sp.Expr );
   ti->setValue( esyn, expr_deletor );   
   p.simplify(2, ti); 
}

}

/* end of parser_dynsym.cpp */
