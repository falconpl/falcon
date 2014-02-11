/*
   FALCON - The Falcon Programming Language.
   FILE: parser_attribute.cpp

   Parser for Falcon source files -- attribute declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_attribute.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>

#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_attribute.h>

#include <falcon/expression.h>
#include <falcon/function.h>
#include <falcon/falconclass.h>

namespace Falcon {

using namespace Parsing;

bool errhand_attribute(const NonTerminal&, Parser& p, int)
{
   TokenInstance* ti = p.getNextToken();

   p.addError( e_syn_attrdecl, p.currentSource(), ti->line(), ti->chr() );
   p.consumeUpTo( p.T_EOL );
   p.clearFrames();
   return true;
}


void apply_attribute( const NonTerminal&, Parser& p )
{
   // T_Colon << T_Name << T_Arrow << Expr << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   p.getNextToken(); // colon
   TokenInstance* tname = p.getNextToken();
   p.getNextToken(); // arrow
   TokenInstance* texpr = p.getNextToken();

   String* name = tname->asString();
   Expression* expr = static_cast<Expression*>(texpr->asData());

   Mantra* mantra = 0;
   if(  ctx->currentFunc() ) {
      mantra = ctx->currentFunc();
   }
   else if( ctx->currentClass() != 0 ) {
      mantra = ctx->currentClass();
   }
   
   bool success = ctx->onAttribute( *name, expr, mantra );

   if( success ) {
      texpr->detachValue();
   }
   else {
      p.addError( e_attrib_already, p.currentSource(), tname->line(), tname->chr() );
   }

   // clear the stack
   p.simplify(5);
}



}

/* end of parser_attribute.cpp */
