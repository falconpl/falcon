/*
   FALCON - The Falcon Programming Language.
   FILE: parser_global.cpp

   Parser for Falcon source files -- Global directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Jan 2013 16:29:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_global.cpp"

#include <falcon/error.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parser_global.h>
#include <falcon/parser/parser.h>
#include <falcon/psteps/stmtglobal.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

bool global_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_global, p.currentSource(), ti->line(), ti->chr() );

   // remove the whole line
   p.consumeUpTo( p.T_EOL );
   p.clearFrames();
   return true;
}


void apply_global( const NonTerminal&, Parser& p )
{
   // << T_global << ListSymbol << T_EOL

   //SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   TokenInstance* tglobal = p.getNextToken();
   TokenInstance* tsyms = p.getNextToken();
   NameList* list = static_cast<NameList*>(tsyms->asData());

   if( ctx->currentFunc() == 0 )
   {
      p.addError( e_global_notin_func, p.currentSource(), tsyms->line(), tsyms->chr() );
   }
   if( list->empty() )
   {
      // global EOL is an error...
      p.addError( e_syn_global, p.currentSource(), tsyms->line(), tsyms->chr() );
   }
   else
   {
      StmtGlobal* global = new StmtGlobal( tglobal->line(), tglobal->chr() );

      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         const String& name = *iter;
         ctx->onGlobal(name);
         if( ! global->addSymbol( name ) )
         {
            p.addError( e_global_again, p.currentSource(), tsyms->line(), tsyms->chr(), 0, name );
         }
         ++iter;
      }

      // add it neverheless
      ctx->currentTree()->append(global);
      ctx->onNewStatement(global);
   }
   
   p.simplify(3);
}

}

/* end of parser_global.cpp */
