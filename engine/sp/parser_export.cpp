/*
   FALCON - The Falcon Programming Language.
   FILE: parser_export.cpp

   Parser for Falcon source files -- export directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 14:31:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_export.cpp"

#include <falcon/error.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/parser_export.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

bool export_errhand(const NonTerminal&, Parser& p, int)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_export, p.currentSource(), ti->line(), ti->chr() );

   p.setErrorMode(&p.T_EOL);
   return true;
}


void apply_export_rule( const NonTerminal&, Parser& p )
{
   // << T_export << ListSymbol << T_EOL 
   
   //SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   p.getNextToken();
   TokenInstance* tsyms = p.getNextToken();
   NameList* list = static_cast<NameList*>(tsyms->asData());

   // if the list is empty, the export is an export all request.
   if( list->empty() )
   {
      ctx->onExport("");
   }
   else
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         ctx->onExport(*iter);
         ++iter;
      }
   }
   
   p.simplify(3);
}

}

/* end of parser_export.cpp */
