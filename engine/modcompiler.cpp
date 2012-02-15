/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.cpp

   Module compiler from non-interactive text file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 16:04:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/error.h>
#include <falcon/modcompiler.h>
#include <falcon/module.h>
#include <falcon/falconclass.h>
#include <falcon/requirement.h>
#include <falcon/vmcontext.h>

#include <falcon/sp/sourcelexer.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprvalue.h>
#include <falcon/importdef.h>

#include "falcon/symbol.h"
#include "falcon/importdef.h"

namespace Falcon {

ModCompiler::Context::Context( ModCompiler* owner ):
   ParserContext( &owner->m_sp ),
   m_owner(owner)
{
}


ModCompiler::Context::~Context()
{

}


void ModCompiler::Context::onInputOver()
{
}


void ModCompiler::Context::onNewFunc( Function* function, Symbol* gs )
{
   Module* mod = m_owner->m_module;
   
   try
   {
      if( gs == 0 )
      {
         // anonymous function
         String name = "__lambda#";
         name.N( m_owner->m_nLambdaCount++);
         function->name( name );
         mod->addFunction( function, false );
      }
      else
      {
         mod->addFunction( gs, function );
      }
   }
   catch( Error* e )
   {
      m_owner->m_sp.addError( e );
      e->decref();
   }
}


void ModCompiler::Context::onNewClass( Class* cls, bool bIsObj, Symbol* gs )
{
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   Module* mod = m_owner->m_module;
   
   try {
      // save the source class
      mod->storeSourceClass( fcls, bIsObj, gs );
   }
   catch( Error* e )
   {
      m_owner->m_sp.addError( e );
      e->decref();
   }
}


void ModCompiler::Context::onNewStatement( Statement* )
{
   // actually nothing to do
}


void ModCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->m_sp;
   
   if( m_owner->m_module->addLoad( path, isFsPath ) == 0 )
   {
      sp.addError( e_load_already, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
   }
}


bool ModCompiler::Context::onImportFrom( ImportDef* def )
{
   if ( ! m_owner->m_module->addImport( def ) )
   {
      SourceParser& sp = m_owner->m_sp;
      sp.addError( e_import_already_mod, sp.currentSource(), sp.currentLine()-1, 0, 0, def->sourceModule() );
      return false;
   }
   
   return true;
}


void ModCompiler::Context::onExport(const String& symName)
{
   Module& mod = *m_owner->m_module;
   SourceParser& sp = m_owner->m_sp;
      
   // Already exporting all?
   if( mod.exportAll() )
   {
      sp.addError( e_export_all, sp.currentSource(), sp.currentLine()-1, 0, 0 );
   }      
   else if( symName == "" )
   {
      mod.exportAll( true );
   }
   else if( symName.startsWith( "_" ) )
   {
      sp.addError( e_export_private, sp.currentSource(), sp.currentLine()-1, 0, 0, symName );
   }
   else {
      bool adef;
      Symbol* sym = mod.addExport( symName, adef );
      if( sym == 0 )
      {
         sp.addError( e_undef_sym, sp.currentSource(), sp.currentLine()-1, 0, 0, symName );
      }
      else if( adef )
      {
         sp.addError( e_export_already, sp.currentSource(), sp.currentLine()-1, 0, 0, symName );
      }
   }
}


void ModCompiler::Context::onDirective(const String&, const String& )
{
   // TODO
}


void ModCompiler::Context::onGlobal( const String& )
{
   // TODO
}


Symbol* ModCompiler::Context::onUndefinedSymbol( const String& name )
{
   Module* mod = m_owner->m_module;
   // Is this a global symbol?
   Symbol* gsym = mod->getGlobal( name );
   if (gsym == 0)
   {
      // no? -- create it as an implicit import.
      gsym = mod->addImplicitImport( name );      
   }
   return gsym;
}


Symbol* ModCompiler::Context::onGlobalDefined( const String& name, bool& bAlreadyDef )
{
   Symbol* sym = m_owner->m_module->getGlobal(name);
   if( sym == 0 )
   {
      bAlreadyDef = false;
      sym = m_owner->m_module->addVariable( name, false );
      sym->declaredAt( m_owner->m_sp.currentLine() );
      return sym;
   }

   bAlreadyDef = true;
   return sym;
}


bool ModCompiler::Context::onUnknownSymbol( const String& uks )
{
   if( ! m_owner->m_module->addImplicitImport( uks ) )
   {
      return false;
   }
   
   return true;
}



Expression* ModCompiler::Context::onStaticData( Class* cls, void* data )
{
   //... which stays alive as long as all the expressions, residing in a function
   // stay alive. The talk may be different for code snippets, but we're dealing
   // with modules here. In short. we have no need for GC.
   return new ExprValue( Item( cls, data ) );
}


void ModCompiler::Context::onRequirement( Requirement* rec )
{
   m_owner->m_module->addRequirement( rec );
}

//=================================================================
//
//

ModCompiler::ModCompiler():
   m_module(0),
   m_nLambdaCount(0),
   m_nClsCount(0)
{
   m_ctx = new Context( this );
   m_sp.setContext( m_ctx );
}

ModCompiler::~ModCompiler()
{
   delete m_ctx;
   delete m_module; // should normally be zero.
}


Module* ModCompiler::compile( TextReader* tr, const String& uri, const String& name )
{
   m_module = new Module( name , uri );
   
   // create the main function that will be used by the compiler
   SynFunc* main = new SynFunc("__main__");
   m_module->addFunction( main, false );

   // and prepare the parser to deal with the main state.
   m_ctx->openMain( &main->syntree() );

   // start parsing.
   m_sp.pushLexer( new SourceLexer( uri, &m_sp, tr ) );      
   if ( !m_sp.parse() )
   {
      // we failed.
      delete m_module;
      return 0;
   }

   // we're done.
   Module* mod = m_module;
   m_module = 0;
   return mod;
}

}

/* end of modcompiler.cpp */
