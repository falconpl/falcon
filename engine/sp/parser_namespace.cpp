/*
   FALCON - The Falcon Programming Language.
   FILE: parser_namespace.cpp

   Parser for Falcon source files -- namespace directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 04 Aug 2011 02:22:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_namespace.cpp"

#include <falcon/error.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parser_namespace.h>

namespace Falcon {

using namespace Parsing;

bool namespace_errhand(const NonTerminal&, Parser& p, int)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_namespace, p.currentSource(), ti->line(), ti->chr() );

   // remove the whole line
   p.setErrorMode(&p.T_EOL);
   return true;
}


void apply_namespace( const NonTerminal&, Parser& p )
{
   // << T_namespace << NameSpaceSpec << T_EOL 
   
   //SourceParser& sp = *static_cast<SourceParser*>(&p);
   //ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();
   String* ns = tname->asString();
   if( ns->find( '*') != String::npos )
   {
      p.addError( e_syn_namespace_star, p.currentSource(), tname->line(), tname->chr() );
   }
   
   static_cast<SourceLexer*>(p.currentLexer())->addNameSpace( *ns );
   
   p.simplify(3);
}

}

/* end of parser_namespace.cpp */
