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

#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_dynsym.h>
#include <falcon/stdhandlers.h>


#include <falcon/engine.h>
#include <falcon/psteps/exprvalue.h>

#include "private_types.h"


namespace Falcon {

using namespace Parsing;


void apply_expr_amper( const NonTerminal&, Parser& p )
{
   static Class* clssym = Engine::handlers()->symbolClass();
   // << T_Amper << T_Name
   SourceParser& sp = static_cast<SourceParser&>(p);

   // get the tokens
   p.getNextToken();
   TokenInstance* tref = p.getNextToken();
   
   // Create the symbol and the reference
   
   // this creates a dynsymbol.
   Symbol* nsym = Engine::getSymbol(*tref->asString() );
   ExprValue* esyn = new ExprValue( Item( clssym, nsym ), tref->line(), tref->chr() );
   
   // update the result token
   TokenInstance* ti = TokenInstance::alloc( tref->line(), tref->chr(), sp.Expr );
   ti->setValue( esyn, treestep_deletor );   
   p.simplify(2, ti); 
}

}

/* end of parser_dynsym.cpp */
