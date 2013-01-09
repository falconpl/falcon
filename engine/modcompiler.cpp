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
   Module* mod = m_owner->m_module;

   class Rator: public Module::MantraEnumerator
   {
   public:
      Rator(Module *mod) : m_mod(mod) {};
      virtual ~Rator() {}
      virtual bool operator()( const Mantra& data, bool )
      {
         if( data.category() == Mantra::e_c_falconclass )
         {
            FalconClass* fcls = const_cast<FalconClass*>(static_cast<const FalconClass*>(&data));

            if( fcls->missingParents() == 0 )
            {
               m_mod->completeClass(fcls);
            }
         }
         return true;
      }
   private:
      Module* m_mod;
   }
   rator(mod);

   mod->enumerateMantras(rator);
}


Variable* ModCompiler::Context::onOpenFunc( Function* function )
{
   Module* mod = m_owner->m_module;
   Variable* var = mod->addMantra( function, false );

   if( ! var == 0  )
   {
      m_owner->m_sp.addError( new CodeError(
         ErrorParam(e_already_def, function->declaredAt(), m_owner->m_module->uri() )
         // TODO add source reference of the imported def
         .symbol( function->name() )
         .origin(ErrorParam::e_orig_compiler)
         ));
   }

   return var;
}


void ModCompiler::Context::onCloseFunc( Function* f )
{
   if( f->name().size() == 0 ) {
      Module* mod = m_owner->m_module;
      mod->addAnonMantra( f );
   }
}


Variable* ModCompiler::Context::onOpenClass( Class* cls, bool )
{
   Module* mod = m_owner->m_module;
   Variable* var = mod->addMantra( cls, false );

   if( ! var )
   {
      m_owner->m_sp.addError( new CodeError(
         ErrorParam(e_already_def, cls->declaredAt(), m_owner->m_module->uri() )
         // TODO add source reference of the imported def
         .origin(ErrorParam::e_orig_compiler)
         ));
   }

   return var;
}


void ModCompiler::Context::onCloseClass( Class* cls, bool )
{
   if( cls->name().size() == 0 ) {
      Module* mod = m_owner->m_module;
      mod->addAnonMantra( cls );
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
      Variable* sym = mod.addExport( symName, adef );
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


Variable* ModCompiler::Context::onGlobalDefined( const String& name, bool& bAlreadyDef )
{
   Variable* var = m_owner->m_module->getGlobal( name );
   if( var == 0 )
   {
      bAlreadyDef = false;
      var = m_owner->m_module->addGlobal( name, Item(), false );
      var->declaredAt( m_owner->m_sp.currentLine() );
   }
   else {
      bAlreadyDef = true;
   }

   return var;
}


Variable* ModCompiler::Context::onGlobalAccessed( const String& name )
{
   Variable* var = m_owner->m_module->getGlobal( name );
   if( var == 0 )
   {
      Variable* var = m_owner->m_module->addImplicitImport( name );
      var->declaredAt( m_owner->m_sp.currentLine() );
   }

   return var;
}


Item* ModCompiler::Context::getVariableValue( Variable* var )
{
   return m_owner->m_module->getGlobalValue( var->id() );
}

void ModCompiler::Context::onRequirement( Requirement* rec )
{
   m_owner->m_module->addRequirement( rec );
}

//=================================================================
//
//

ModCompiler::ModCompiler():
   m_module(0)
{
   m_ctx = new Context( this );
   m_sp.setContext( m_ctx );
}

ModCompiler::ModCompiler( ModCompiler::Context* ctx ):
   m_module(0)
{
   m_ctx = ctx;
   m_sp.setContext( m_ctx );
}

ModCompiler::~ModCompiler()
{
   delete m_ctx;
   if( m_module != 0 )
   {
      m_module->decref();
   }
}


Module* ModCompiler::compile( TextReader* tr, const String& uri, const String& name )
{
   m_module = new Module( name , uri, false ); // not a native module

   // create the main function that will be used by the compiler
   SynFunc* main = new SynFunc("__main__");

   // and prepare the parser to deal with the main state.
   m_ctx->openMain( &main->syntree() );

   // start parsing.
   m_sp.pushLexer( new SourceLexer( uri, &m_sp, tr ) );
   if ( !m_sp.parse() )
   {
      // we failed.
      m_module->decref();
      return 0;
   }

   if(! main->syntree().empty() ) {
      m_module->setMainFunction( main );
   }
   else {
      delete main;
   }

   // we're done.
   Module* mod = m_module;
   m_module = 0;
   return mod;
}

}

/* end of modcompiler.cpp */
