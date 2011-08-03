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

/** Internal class representing an import as or in clause*/
class ImportClause
{
public:
   String* m_target;
   bool m_bIsIn;
   
   ImportClause( String* str, bool b ):
      m_target( str ),
      m_bIsIn( b )
   {}
   
   ~ImportClause() {
      delete m_target;
   }
};


static void import_clause_deletor( void* data )
{
   delete static_cast<ImportClause*>(data);
}


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


static void apply_import_internal( Parser&p, 
   TokenInstance* tnamelist,  TokenInstance* tdepName, 
   TokenInstance* importClause, bool bNameIsPath )
{
   ParserContext* ctx = static_cast<ParserContext*>( p.context() );
   
   NameList* list = static_cast<NameList*>(tnamelist->asData());
   String* name = tdepName->asString();
   ImportClause* ic = static_cast<ImportClause*>(importClause->asData());
   
   if( list->empty() )
   {
      if( ic == 0 )
      {
         // it's a general import/from.
         // general import-from don't support as.
         if( ! ic->m_bIsIn )
         {
            p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
         }

         // import from modname
         // TODO: use ic->m_target just to create the namespace in the compiler,
         // As import from Module in ... is the same as import from Module to us.
         // we don't need to know 
         ctx->onImportFrom( *name, bNameIsPath, "", "", false );
      }
   }
   else
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // this will eventually add the needed errors.
         if( ic != 0 )
         {
            ctx->onImportFrom( *name, bNameIsPath, *iter, *ic->m_target, ic->m_bIsIn );
         }
         else
         {
            ctx->onImportFrom( *name, bNameIsPath, *iter, "", false );
         }
         ++iter;
         
         // as clause and more things?
         if( ic != 0 && ! ic->m_bIsIn && iter != list->end() )
         {
            // import/from/as supports only one symbol
            p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
            break;
         }
      }      
   }
   
   // Declaret that this namelist is a valid ImportClause
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   importClause->token( sp.ImportClause );
   
   // let the last token to live this way.
   p.simplify( 3 );
}


void apply_import_from_in( const Rule&, Parser& p )
{
   // << NameList << T_from << T_String << ImportFromInClause );
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken(); // T_From
   TokenInstance* tdepName = p.getNextToken();
   TokenInstance* importClause = p.getNextToken();
   
   apply_import_internal( p, tnamelist,  tdepName, importClause, true );
}


void apply_import_from_in_modspec( const Rule&, Parser& p )
{
   // << NameList << T_from << T_ModSpec << ImportFromInClause );
   TokenInstance* tnamelist = p.getNextToken();
   p.getNextToken(); // T_From
   TokenInstance* tdepName = p.getNextToken();
   TokenInstance* importClause = p.getNextToken();
   
   apply_import_internal( p, tnamelist,  tdepName, importClause, false );
}


void apply_import_namelist( const Rule&, Parser& p )
{
   //<< NameList
   
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
   tnamelist->token( sp.ImportClause );
   // nothing to simplify
}


static void apply_import_fromin_internal( Parser& p, bool bInAs )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   
   p.getNextToken();
   TokenInstance* tmodspec = p.getNextToken();
   
   TokenInstance* tic = new TokenInstance( tmodspec->line(), tmodspec->chr(), 
                                                      sp.ImportFromInClause );
      
   tic->setValue( new ImportClause( tmodspec->detachString(), bInAs ), import_clause_deletor );
   p.simplify(3, tic);   
}


void apply_import_fromin( const Rule&, Parser& p )
{
   apply_import_fromin_internal( p, true );
}


void apply_import_fromas( const Rule&, Parser& p )
{
   // << T_as << Name << T_EOL
   apply_import_fromin_internal( p, false );
}


void apply_import_from_empty( const Rule&, Parser& p )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   p.getNextToken()->token(sp.ImportFromInClause);
}

}

/* end of parser_import.cpp */
