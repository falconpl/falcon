/*
   FALCON - The Falcon Programming Language.
   FILE: parser_load.cpp

   Parser for Falcon source files -- index accessor handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 14:31:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_load.cpp"

#include <falcon/error.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/parser_load.h>

namespace Falcon {

using namespace Parsing;

bool load_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser* sp = static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_load, p.currentSource(), ti->line(), ti->chr() );

   // remove the whole line
   // TODO: Sync with EOL
   p.clearFrames();
   return true;
}


void apply_load_string( const Rule&, Parser& p )
{
   //SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   
   // T_load << T_String << T_EOL
   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   ctx->onLoad( *ti->asString(), true );
   p.simplify(3);
}

void apply_load_mod_spec( const Rule&, Parser& p )
{
   // T_load << ModSpec << T_EOL
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   p.getNextToken();
   TokenInstance* ti = p.getNextToken();
   ctx->onLoad( *ti->asString(), false );
   p.simplify(3);
}


bool load_modspec_errhand(const NonTerminal& nt, Parser& p)
{
   // for now, jsut the same.
   return load_errhand( nt, p );
}

void apply_modspec_next( const Rule&, Parser& p )
{
   //ModSpec << T_dot << T_Name
   SourceParser* sp = static_cast<SourceParser*>(&p);
   
   TokenInstance* tspec = p.getNextToken();
   p.getNextToken();
   TokenInstance* tspec_cont = p.getNextToken();
   
   TokenInstance* tspec_new = TokenInstance::alloc( tspec->line(), tspec->chr(), sp->ModSpec );
   // put the cumulated spec in the last token.
   tspec_new->setValue( *tspec->asString() + "." + *tspec_cont->asString() );
   p.simplify(3, tspec_new);
}


void apply_modspec_first( const Rule&, Parser& p )
{
   // T_Name
   SourceParser* sp = static_cast<SourceParser*>(&p);
   TokenInstance* tfirst = p.getNextToken();
   TokenInstance* tspec = TokenInstance::alloc( tfirst->line(), tfirst->chr(), sp->ModSpec );
   tspec->setValue( *tfirst->asString() );
   
   p.simplify(1, tspec);
}

void apply_modspec_first_self( const Rule&, Parser& p)
{
   // T_Name
   SourceParser* sp = static_cast<SourceParser*>(&p);
   TokenInstance* tfirst = p.getNextToken();
   TokenInstance* tspec = TokenInstance::alloc( tfirst->line(), tfirst->chr(), sp->ModSpec );
   tspec->setValue( String("self") );
   
   p.simplify(1, tspec);
}

void apply_modspec_first_dot( const Rule&, Parser& p)
{
   // T_Dot
   SourceParser* sp = static_cast<SourceParser*>(&p);
   TokenInstance* tfirst = p.getNextToken();
   TokenInstance* tname = p.getNextToken();
   TokenInstance* tspec = TokenInstance::alloc( tfirst->line(), tfirst->chr(), sp->ModSpec );
   tspec->setValue( String("." + *tname->asString()) );
   p.simplify(2, tspec);
}

}

/* end of parser_load.cpp */
