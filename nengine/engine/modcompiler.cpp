/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.cpp

   Interactive compiler
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
#include <falcon/hyperclass.h>
#include <falcon/exprvalue.h>
#include <falcon/globalsymbol.h>
#include <falcon/unknownsymbol.h>
#include <falcon/inheritance.h>
#include <falcon/requirement.h>

#include <falcon/sp/sourcelexer.h>
#include <falcon/parser/parser.h>


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
   // todo: check undefined things.
}

void ModCompiler::Context::onNewFunc( Function* function, GlobalSymbol* gs )
{
   if( gs == 0 )
   {
      // anonymous function
      String name = "__lambda#";
      name.N( m_owner->m_nLambdaCount++);
      function->name( name );
      m_owner->m_module->addFunction( function, false );
   }
   else
   {
      m_owner->m_module->addFunction( gs, function );
   }
}


void ModCompiler::Context::onNewClass( Class* cls, bool bIsObj, GlobalSymbol* gs )
{
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   m_owner->m_module->storeSourceClass( fcls, bIsObj, gs );
}


void ModCompiler::Context::onNewStatement( Statement* )
{
   // actually nothing to do
}


void ModCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_module->addLoad( path, isFsPath ) )
   {
      sp.addError( e_load_already, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
   }
}


void ModCompiler::Context::onImportFrom( const String& path, bool isFsPath, const String& symName,
      const String& nsName, bool bIsNS )
{
   if( nsName != "" )
   {
      if( bIsNS )
      {        
         m_owner->m_module->addImportFromWithNS( nsName, symName, path, isFsPath );
      }
      else
      {
         // it's an as -- and a pure one, as the parser removes possible "as" errors.
         m_owner->m_module->addImportFrom( nsName, symName, path, isFsPath );
      }
   }
   else
   {
      if( symName == "" )
      {
         if( ! m_owner->m_module->addGenericImport( path, isFsPath ) )
         {
            SourceParser& sp = m_owner->m_sp;
            sp.addError( e_import_already_mod, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
         }
      }
      else
      {
         if( symName.endsWith("*") )
         {
            // fake "import a.b.c.* from Module in a.b.c
            m_owner->m_module->addImportFromWithNS( 
                        symName.length() > 2 ? 
                                 symName.subString(0, symName.length()-2) : "", 
                        symName, path, isFsPath );
         }
         else
         {
            m_owner->m_module->addImportFrom( symName, symName, path, isFsPath );
         }
      }
   }  
}


void ModCompiler::Context::onImport(const String& symName )
{
   if( symName.endsWith("*") )
   {
      if( symName.length() == 1 )
      {
         SourceParser& sp = m_owner->m_sp;
         sp.addError( e_syn_import, sp.currentSource(), sp.currentLine()-1, 0, 0, symName );
      }
      else
      {
         // fake "import a.b.c.* from "" in a.b.c
         m_owner->m_module->addImportFromWithNS( 
                  symName.subString(0, symName.length()-2), 
                  symName, "", true );
      }
   }
   else if( ! m_owner->m_module->addImport( symName ) )
   {
      SourceParser& sp = m_owner->m_sp;
      sp.addError( e_import_already, sp.currentSource(), sp.currentLine()-1, 0, 0, symName );
   }
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
   // Is this a global symbol?
   Symbol* gsym = m_owner->m_module->getGlobal( name );
   // --- if not, let the onUnknownSymbol to be called for implicit import.
   return gsym;
}


GlobalSymbol* ModCompiler::Context::onGlobalDefined( const String& name, bool& bAlreadyDef )
{
   Symbol* sym = m_owner->m_module->getGlobal(name);
   if( sym == 0 )
   {
      bAlreadyDef = false;
      return m_owner->m_module->addVariable( name, false );
   }

   bAlreadyDef = true;
   return static_cast<GlobalSymbol*>(sym);
}


bool ModCompiler::Context::onUnknownSymbol( UnknownSymbol* uks )
{
   if( ! m_owner->m_module->addImplicitImport( uks ) )
   {
      delete uks;
      return false;
   }
   
   return true;
}



Expression* ModCompiler::Context::onStaticData( Class* cls, void* data )
{
   // The data resides in the module...
   m_owner->m_module->addStaticData( cls, data );

   //... which stays alive as long as all the expressions, residing in a function
   // stay alive. The talk may be different for code snippets, but we're dealing
   // with modules here. In short. we have no need for GC.
   return new ExprValue( Item( cls, data ) );
}


void ModCompiler::Context::onInheritance( Inheritance* inh  )
{
   // In the interactive compiler context, classes must have been already defined...
   const Symbol* sym = m_owner->m_module->getGlobal(inh->className());
  
   // found?
   if( sym != 0 )
   {
      Item* itm;
      if( ( itm = sym->value() ) == 0 )
      {
         m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      // we want a class. A rdeal class.
      else if ( ! itm->isUser() || ! itm->asClass()->isMetaClass() )
      {
         m_owner->m_sp.addError( e_inv_inherit, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      else
      {
         inh->parent( static_cast<Class*>(itm->asInst()) );
      }

      // ok, now how to break away if we aren't complete?
   }
   else
   {
      m_owner->m_module->addImportInheritance( inh );
   }
}

void ModCompiler::Context::onRequirement( Requirement* rec )
{
   Error* error = 0;
   // In the interactive compiler context, classes must have been already defined...
   const Symbol* sym = m_owner->m_module->getGlobal(rec->name());
   
   if( sym != 0 )
   {
      error = rec->resolve( 0, sym );
      if( error != 0 )
      {
         // The error is a bout a local symbol. We can "downgrade" it.
         m_owner->m_sp.addError( error->errorCode(), error->module(), error->line(), 0 );
         error->decref();
      }
   }
   else
   {
      // otherwise, just let the requirement pending in the module.
      m_owner->m_module->addRequirement( rec );
   }
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
