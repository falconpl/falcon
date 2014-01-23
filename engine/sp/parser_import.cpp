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

#include <falcon/error.h>
#include <falcon/importdef.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/parser_import.h>
#include <falcon/sp/sourcelexer.h>

#include <list>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

bool import_errhand(const NonTerminal&, Parser& p)
{
   //SourceParser& sp = *static_cast<SourceParser*>(p);
   TokenInstance* ti = p.getNextToken();
   
   // already detected?
   if( p.lastErrorLine() != ti->line() )
   {
      p.addError( e_syn_import, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // remove the whole line
   p.consumeUpTo( p.T_EOL );
   p.clearFrames();
   return true;
}


void apply_import( const Rule&, Parser& p )
{
   //<< T_import << ImportClause << T_EOL  );
   //we have already applied the import
   p.simplify(3);
}


static void apply_import_star( Parser& p, const String& modspec, bool bIsPath, const String& ns )
{      
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>( p.context() );
   ImportDef* id = new ImportDef(modspec, bIsPath, "*", ns, true);
   id->sr().line( sp.currentLine() );
   
   if( ctx->onImportFrom( id ) && ns != "" )
   {
      static_cast<SourceLexer*>(p.currentLexer())->addNameSpace( ns );
   }
   
   // leave the import clause
   TokenInstance* tEOL = p.getLastToken();
   tEOL->token( sp.ImportClause );
   p.simplify( ns == "" ? 3 : 5 );
}


static void apply_import_internal( Parser& p, 
   TokenInstance* tnamelist,  
   TokenInstance* tdepName, bool bNameIsPath,
   TokenInstance* tInOrAs,  bool bIsIn )
{
   NameList* list = static_cast<NameList*>(tnamelist->asData());
   String* name = tdepName->asString();
   String* nspace = tInOrAs == 0 ? 0 : tInOrAs->asString();   
   
   // -- we might have several failure reasons
   bool bOk = true;
   
   // place where to store namespaces that we're finding.
   // -- we must check that the import is OK before applying them.
   NameList nsList;
   // The forming import definition; we may need to delete it on error.
   ImportDef* def = 0;
   
   if( nspace != 0 && nspace->find( '*') != String::npos )
   {
      // notice that "as" grammar cannot generate a "*" here.
      p.addError( e_syn_namespace_star, p.currentSource(), tdepName->line(), tdepName->chr() );
      bOk = false;
   }
   else
   {
      def = new ImportDef;      
      def->setImportModule( *name, bNameIsPath );
      def->sr().line( p.currentLine() );
      
      if (nspace != 0)
      {
         if( bIsIn )
         {
            def->setTargetNS( *nspace );
            nsList.push_back( *nspace );
         }
         else
         {
            if( list->size() != 1 )
            {
               bOk = false;
               p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
            }
            else
            {
               def->setTargetSymbol( *nspace );
            }
         }
      }
   }
   
   if( bOk )
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         const String& symName = *iter;
         // accept asterisk in names only at the end.
         length_t starPos = symName.find( '*' );
         if( starPos != String::npos && starPos < symName.length()-1 )
         {
            bOk = false;
            p.addError( e_syn_import_name_star, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
            break;
         }
         
         // this will eventually add the needed errors.
         if( tInOrAs != 0 )
         {
            if( ! bIsIn && starPos == symName.length()-1 )
            {
               // as cannot be used with *
               bOk = false;
               p.addError( e_syn_import_as, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
               break; // force loop break, "as" accept just one symbol and we had already one error.
            }
            else
            {
               def->addSourceSymbol( symName );
            }
         }
         else
         {
            def->addSourceSymbol( symName );
            
            // check implicit namespace creation.
            if( (starPos = symName.rfind( '.')) != String::npos )
            {               
               nsList.push_back(symName.subString(0, starPos ) );
            }
         }
         ++iter;        
      }      
   }
   
   // If all is ok, apply what we have done.
   if( bOk )
   {
      // does the import definition conflicts with the module?
      ParserContext* ctx = static_cast<ParserContext*>( p.context() );
      if( ctx->onImportFrom( def ) )
      {
         // ok, we can apply the namespaces.
         NameList::iterator iter = nsList.begin();
         while( iter != nsList.end() )
         {
            static_cast<SourceLexer*>(p.currentLexer())->addNameSpace( *iter );
            ++iter;
         }
      }
      else {
         delete def;
      }
   }
   else
   {
      // delete what we have partially done.
      delete def;
   }
      
   // Declaret that this namelist is a valid ImportClause
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   TokenInstance* ti = p.getLastToken();
   ti->token( sp.ImportClause );
   
   // let the last token to live this way.
   p.simplify( tInOrAs == 0 ? 3 : 5 );
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


void apply_import_star_from_string_in( const Rule&, Parser& p )
{
   // << T_Times << T_from << T_String << T_in << T_Name << T_EOL
   p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_star( p, *tdepName->asString(), true, *tInOrAs->asString() );
}

void apply_import_star_from_string( const Rule&, Parser& p )
{
   // << T_Times << T_from << T_String << T_EOL
   p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   
   apply_import_star( p, *tdepName->asString(), true, "" );
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


void apply_import_star_from_modspec_in( const Rule&, Parser& p )
{
   // << T_Times << T_from << ModSpec << T_in << T_Name << T_EOL
   p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   p.getNextToken();
   TokenInstance* tInOrAs = p.getNextToken();
   
   apply_import_star( p, *tdepName->asString(), false, *tInOrAs->asString() );
}


void apply_import_star_from_modspec( const Rule&, Parser& p )
{
   // << T_Times << T_from << T_String << T_EOL
   p.getNextToken();
   p.getNextToken();
   TokenInstance* tdepName = p.getNextToken();
   
   apply_import_star( p, *tdepName->asString(), false, "" );
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
   // << ImportSpec << T_EOL
   
   TokenInstance* tnamelist = p.getNextToken();
   NameList* list = static_cast<NameList*>(tnamelist->asData());
   ParserContext* ctx = static_cast<ParserContext*>( p.context() );
   
   ImportDef* id = new ImportDef;
   id->sr().line( p.currentLine() );
   
   NameList nspaces;
   bool bOk = true;
   
   if( list->empty() )
   {
      // there can't be an empty list at this point.
      p.addError( e_syn_import, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
      bOk = false;
   }
   else
   {
      NameList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // this will eventually add the needed errors.
         // check for "*"
         const String& symName = *iter;
         length_t starPos = symName.find( '*' );
         if( starPos != String::npos && starPos < symName.length()-1 )
         {
            p.addError( e_syn_import_name_star, p.currentSource(), tnamelist->line(), tnamelist->chr(), 0 );
            bOk = false;
         }         
         else
         {
            // check implicit namespace creation.
            if( starPos == symName.length()-1 )
            {
               nspaces.push_back( symName.subString(0, symName.length()-2 ) );
            }
            
            id->addSourceSymbol( symName );
         }
         
         ++iter;
      }
   }

   if( bOk )
   {
      bOk = ctx->onImportFrom( id );
      if ( bOk )
      {
         NameList::iterator nli = nspaces.begin();
         while( nli != nspaces.end() )
         {
            static_cast<SourceLexer*>(p.currentLexer())->
                  addNameSpace( *nli );
            ++nli;
         }
      }
   }
   
   if ( ! bOk )
   {
      delete id;
   }
   
   // Declare that this namelist is a valid ImportClause
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   /*p.getNextToken()->token( sp.ImportClause ); // turn the EOL in importclause
   // and remove the list token.
   p.simplify(1,0);
   */
   tnamelist->token(sp.ImportClause);
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

   TokenInstance* ti_list = TokenInstance::alloc(tlist->line(), tlist->chr(), sp.ImportSpec );
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

   TokenInstance* ti_list = TokenInstance::alloc(tlist->line(), tlist->chr(), sp.ImportSpec );
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

   TokenInstance* ti_list = TokenInstance::alloc(tlist->line(), tlist->chr(), sp.ImportSpec );
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 3, ti_list );   
}


void apply_ImportSpec_first( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tname = p.getNextToken();

   TokenInstance* ti_list = TokenInstance::alloc(tname->line(), tname->chr(), sp.ImportSpec );

   NameList* list = new NameList;
   list->push_back(*tname->asString());
   ti_list->setValue( list, name_list_deletor );

   p.simplify( 1, ti_list );   
}


void apply_ImportSpec_empty( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   //TODO: Get current lexer char/line
   TokenInstance* ti_list = TokenInstance::alloc(0, 0, sp.ImportSpec );

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
   TokenInstance* tspec_new = TokenInstance::alloc( tspec->line(), tspec->chr(), sp->NameSpaceSpec );
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
   
   TokenInstance* tspec_new = TokenInstance::alloc( tspec->line(), tspec->chr(), sp->NameSpaceSpec );
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
