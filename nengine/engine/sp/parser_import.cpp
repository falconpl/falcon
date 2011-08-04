/*
   FALCON - The Falcon Programming Language.
   FILE: parser_import.cpp

   Parser for Falcon source files -- import/from directive
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 02 Aug 2011 20:50:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_import.cpp"

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/parser_import.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

bool import_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser& sp = *static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_import, p.currentSource(), ti->line(), ti->chr() );

   // remove the whole line
   // TODO: Sync with EOL
   p.clearFrames();
   return true;
}


void apply_import( const Rule&, Parser& p )
{
   //<< T_import << ImportClause  );
   //we have already applied the import
   p.simplify(2);
}


static void apply_import_internal( Parser& p, 
   TokenInstance* tnamelist,  
   TokenInstance* tdepName, bool bNameIsPath,
   TokenInstance* tInOrAs,  bool bIsIn )
{
   ParserContext* ctx = static_cast<ParserContext*>( p.context() );
   
   NameList* list = static_cast<NameList*>(tnamelist->asData());
   String* name = tdepName->asString();
   
   if( list->empty() )
   {
      if( tInOrAs == 0 )
      {         
         // "import from modname"
         ctx->onImportFrom( *name, bNameIsPath, "", "", false );
      }
      else
      {
         // TODO: use ic->m_target just to create the namespace in the compiler,
         // it's a general import/from in ns.
         if( bIsIn )
         {
            ctx->onImportFrom( *name, bNameIsPath, "", *tInOrAs->asString(), true );
         }
         else
         {
            // general import-from don't support as.
            p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
         }
      }
   }
   else
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // this will eventually add the needed errors.
         if( tInOrAs != 0 )
         {
            ctx->onImportFrom( *name, bNameIsPath, *iter, *tInOrAs->asString(), bIsIn );
         }
         else
         {
            ctx->onImportFrom( *name, bNameIsPath, *iter, "", false );
         }
         ++iter;
         
         // as clause and more things?
         if( tInOrAs != 0 && ! bIsIn && iter != list->end() )
         {
            // import/from/as supports only one symbol
            p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
            break;
         }
      }      
   }
   
   // Declaret that this namelist is a valid ImportClause
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   TokenInstance* ti = new TokenInstance( 
                     tnamelist->line(), tnamelist->chr(), sp.ImportClause );
   
   // let the last token to live this way.
   p.simplify( tInOrAs == 0 ? 4 : 6, ti );
}


void apply_import_from_string_as( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_as << T_Name << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_internal( p, tnamelist, 
      tdepName, true,
      tInOrAs, false );
}

void apply_import_from_string_in( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_in << T_Name << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_internal( p, tnamelist, 
      tdepName, true,
      tInOrAs, true );
}


void apply_import_string( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();   
   
   apply_import_internal( p, tnamelist, 
      tdepName, true,
      0, false );
}


void apply_import_from_modspec_as( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_as << T_Name << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_internal( p, tnamelist, 
      tdepName, false,
      tInOrAs, false );
}

void apply_import_from_modspec_in( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_in << T_Name << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_internal( p, tnamelist, 
      tdepName, false,
      tInOrAs, true );
}


void apply_import_from_modspec( const Rule&, Parser& p )
{
   // << ListSymbol << T_from << T_String << T_EOL
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();   
   
   apply_import_internal( p, tnamelist, 
      tdepName, false,
      0, false );
}

void apply_import_syms( const Rule&, Parser& p )
{
   // << ListSymbol << T_EOL
   
   TokenInstance* tnamelist = p.getNextToken();
   NameList* list = static_cast<NameList*>(tnamelist->asData());
   ParserContext* ctx = static_cast<ParserContext*>( p.context() );
   
   if( list->empty() )
   {
      // there can't be an empty list at this point.
      p.addError( e_syn_import, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
   }
   else
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // this will eventually add the needed errors.
         ctx->onImport(*iter);
         ++iter;
      }
   }
   
   // Declaret that this namelist is a valid ImportClause
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   p.getNextToken()->token( sp.ImportClause ); // turn the EOL in importclause
   // and remove the list token.
   p.simplify(1,0);
}


//===================================================================
//

bool importspec_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser& sp = *static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   p.addError( e_syn_import_spec, p.currentSource(), ti->line(), ti->chr() );

   // remove the whole line
   // TODO: Sync with EOL
   p.clearFrames();
   return true;
}

void apply_ImportSpec_next( const Rule&, Parser& p )
{
   //  ImportSpec << T_Comma << NameSpaceSpec )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->detachValue());
   list->push_back(*tname->asString());

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ImportSpec );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );   
}

void apply_ImportSpec_attach_last( const Rule&, Parser& p )
{
   //  ImportSpec << T_Dot << T_Times )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->detachValue());
   list->back().A(".*");

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ImportSpec );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );  
}


void apply_ImportSpec_attach_next( const Rule&, Parser& p )
{
   //  ImportSpec << T_Dot << NameSpaceSpec )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tlist = p.getNextToken();
   p.getNextToken();
   TokenInstance* tname = p.getNextToken();

   NameList* list=static_cast<NameList*>(tlist->detachValue());
   list->back().A(".").A(*tname->asString());

   TokenInstance* ti_list = new TokenInstance(tlist->line(), tlist->chr(), sp.ImportSpec );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );   
}


void apply_ImportSpec_first( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   TokenInstance* ti_list = new TokenInstance(tname->line(), tname->chr(), sp.ImportSpec );

   NameList* list = new NameList;
   list->push_back(*tname->asString());
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 1, ti_list );   
}


void apply_ImportSpec_empty( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = new TokenInstance(0, 0, sp.ImportSpec );

   NameList* list = new NameList;
   ti_list->setValue( list, name_list_deletor );

   // Nothing to delete, just insert.
   p.simplify( 0, ti_list );
}


void apply_nsspec_last( const Rule&, Parser& p )
{
   //NameSpaceSpec << T_dot << T_Times
   SourceParser* sp = static_cast<SourceParser*>(&p);
   
   TokenInstance* tspec = p.getNextToken();
   TokenInstance* tspec_new = new TokenInstance( tspec->line(), tspec->chr(), sp->NameSpaceSpec );
   // put the cumulated spec in the last token.
   tspec_new->setValue( *tspec->asString() + ".*" );
   p.simplify(3, tspec_new);
}


void apply_nsspec_next( const Rule&, Parser& p )
{
   //NameSpaceSpec << T_dot << T_Name
   SourceParser* sp = static_cast<SourceParser*>(&p);
   
   TokenInstance* tspec = p.getNextToken();
   p.getNextToken();
   TokenInstance* tspec_cont = p.getNextToken();
   
   TokenInstance* tspec_new = new TokenInstance( tspec->line(), tspec->chr(), sp->NameSpaceSpec );
   // put the cumulated spec in the last token.
   tspec_new->setValue( *tspec->asString() + "." + *tspec_cont->asString() );
   p.simplify(3, tspec_new);
}


void apply_nsspec_first( const Rule&, Parser& p )
{
   // T_Name
   SourceParser* sp = static_cast<SourceParser*>(&p);
   TokenInstance* tfirst = p.getNextToken();
   tfirst->token( sp->NameSpaceSpec );   
}



}

/* end of parser_import.cpp */
