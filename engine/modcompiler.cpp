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
#include <falcon/vmcontext.h>
#include <falcon/synclasses_id.h>

#include <falcon/sp/sourcelexer.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprlit.h>
#include <falcon/importdef.h>
#include <falcon/stderrors.h>

#include <falcon/symbol.h>
#include <falcon/importdef.h>

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
   SourceParser& sp = m_owner->m_sp;
   ParserContext* pctx = static_cast<ParserContext*>(sp.context());
   if( pctx->currentFunc() != 0
            || pctx->currentLitContext() != 0
            || pctx->currentStmt() != 0
            || pctx->currentClass() != 0
            )
   {
      String error;
      int line = 0;
      if( pctx->currentFunc() != 0 )
      {
         error = "Function not closed ";
         line = pctx->currentFunc()->declaredAt();
      }
      else if( pctx->currentLitContext() != 0 )
      {
         error = "LiteralContext not closed";
         line = pctx->currentLitContext()->sr().line();
      }
      else if( pctx->currentStmt() != 0 )
      {
         error = "Context not closed";
         line = pctx->currentStmt()->sr().line();
      }
      else if( pctx->currentClass() != 0 )
      {
         error = "Class not closed";
         line = pctx->currentClass()->declaredAt();
      }
      else
      {
         error = "Element not closed";
         line = sp.lastLine();
      }

      sp.addError(e_runaway_eof, sp.lastSource(), sp.lastLine(), 0, line, error );
   }

   /*
   Module* mod = m_owner->m_module;

   class Rator: public Module::MantraEnumerator
   {
   public:
      Rator(Module *mod) : m_mod(mod) {};
      virtual ~Rator() {}
      virtual bool operator()( const Mantra& data )
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
   */
}


bool ModCompiler::Context::onOpenFunc( Function* function )
{
   Module* mod = m_owner->m_module;
   if( function->name().size() == 0 ) {
      mod->addAnonMantra( function );
      return false;
   }
   else
   {
      bool ok = mod->addMantra( function, false );

      if( ! ok )
      {
         // save it somewhere anyhow
         mod->addAnonMantra( function );

         m_owner->m_sp.addError( new CodeError(
            ErrorParam(e_already_def, function->declaredAt(), m_owner->m_module->uri() )
            // TODO add source reference of the imported def
            .symbol( function->name() )
            .origin(ErrorParam::e_orig_compiler)
            ));
      }

      return ok;
   }
}


void ModCompiler::Context::onOpenMethod( Class* cls, Function* function )
{
   Module* mod = m_owner->m_module;

   if( static_cast<FalconClass*>(cls)->getProperty(function->name()) != 0 )
   {
      m_owner->m_sp.addError( new CodeError(
             ErrorParam(e_prop_adef, function->declaredAt(), m_owner->m_module->uri() )
             // TODO add source reference of the imported def
             .symbol( function->name() )
             .origin(ErrorParam::e_orig_compiler)
             ));
   }
   function->module(mod);
}


void ModCompiler::Context::onCloseFunc( Function* )
{

}


bool ModCompiler::Context::onOpenClass( Class* cls, bool isObject )
{
   Module* mod = m_owner->m_module;
   bool status;
   cls->module(mod);

   if ( isObject )
   {
      // add a global as a placeholder for the object
      status = mod->addInitClass(cls, false);
   }
   else {
      if( cls->name().size() == 0 ) {
         Module* mod = m_owner->m_module;
         mod->addAnonMantra( cls );
         return 0;
      }
      else {
         status = mod->addMantra( cls, false );
      }
   }

   if( ! status )
   {
      // save it anyhow
      mod->addAnonMantra( cls );

      m_owner->m_sp.addError( new CodeError(
         ErrorParam(e_already_def, cls->declaredAt(), m_owner->m_module->uri() )
         // TODO add source reference of the imported def
         .origin(ErrorParam::e_orig_compiler)
         ));
   }

   return status;
}

bool ModCompiler::Context::onAttribute(const String& name, TreeStep* generator, Mantra* target )
{
   Module* mod = m_owner->m_module;
   Attribute* added = 0;

   if ( target == 0 )
   {
      added = mod->attributes().add(name);
   }
   else {
      added = target->attributes().add(name);
   }

   if( added != 0 ) {
      added->generator(generator);
      return true;
   }

   return false;
}


void ModCompiler::Context::onCloseClass( Class*, bool )
{
}



void ModCompiler::Context::onNewStatement( TreeStep* ts )
{
   if( ts->category() == TreeStep::e_cat_expression )
   {
      Expression* expr = static_cast<Expression*>(ts);
      if( ! expr->isStandAlone() )
      {
         // are we in a lambda?
         ParserContext* pc =  static_cast<ParserContext*>(m_owner->m_sp.context());
         if((pc->currentFunc() != 0 && (pc->currentFunc()->name().size() == 0|| pc->currentFunc()->name().startsWith("_anon#") ))
             || pc->isLitContext()
             || (pc->currentStmt() != 0 && pc->currentStmt()->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE ) )
         {
            // it's ok
         }
         else {
            SourceParser& sp = m_owner->m_sp;
            sp.addError( e_noeffect, sp.currentSource(), ts->line(), ts->chr(), 0 );
         }
      }
   }
}


void ModCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->m_sp;

   Error* err = 0;
   m_owner->m_module->addLoad( path, isFsPath, err );
   if( err )
   {
      sp.addError( err );
   }
}


bool ModCompiler::Context::onImportFrom( ImportDef* def )
{
   Error* err = 0;
   m_owner->m_module->addImport( def, err );
   if( err ) {
      SourceParser& sp = m_owner->m_sp;
      sp.addError( err );
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
      bool adef = false;

      if( ! mod.globals().exportGlobal(symName, adef) )
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


void ModCompiler::Context::onGlobal( const String& name )
{
   // global defines the global variable if not already there.
   GlobalsMap::Data* var = m_owner->m_module->globals().get( name );

   if( var == 0 )
   {
      var = m_owner->m_module->globals().add( name, Item(), false );
   }
}


void ModCompiler::Context::onGlobalDefined( const String& name, bool& bAlreadyDef )
{
   GlobalsMap::Data* var = m_owner->m_module->globals().get( name );
   if( var == 0 )
   {
      bAlreadyDef = false;
      m_owner->m_module->globals().add( name, Item(), false );
   }
   else {
      bAlreadyDef = true;
   }
}


bool ModCompiler::Context::onGlobalAccessed( const String& name )
{
   GlobalsMap::Data* var = m_owner->m_module->globals().get( name );

   if( var == 0 && ! m_owner->sp().interactive() )
   {
      m_owner->m_module->addImplicitImport( name, m_owner->m_sp.currentLine() );
   }

   return var != 0 && ! var->m_bExtern;
}


Item* ModCompiler::Context::getVariableValue( const String& name )
{
   return m_owner->m_module->globals().getValue( name );
}


void ModCompiler::Context::onIString(const String& string )
{
   m_owner->m_module->addIString( string );
}

Item* ModCompiler::Context::getValue( const Symbol* sym )
{
   return m_owner->m_module->resolve( sym );
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


Module* ModCompiler::compile( TextReader* tr, const String& uri, const String& name, bool asFtd )
{
   m_module = new Module( name , uri, false ); // not a native module

   // create the main function that will be used by the compiler
   SynFunc* main = new SynFunc("__main__");

   // and prepare the parser to deal with the main state.
   m_ctx->openMain( &main->syntree() );

   // start parsing.
   SourceLexer* slex = new SourceLexer( uri, &m_sp, tr );
   slex->setParsingFam(asFtd);
   m_sp.pushLexer( slex );
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
