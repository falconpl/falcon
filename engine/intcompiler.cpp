/*
   FALCON - The Falcon Programming Language.
   FILE: intcompiler.cpp

   Interactive compiler - step by step dynamic compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/intcompiler.cpp"

#include <falcon/intcompiler.h>
#include <falcon/module.h>
#include <falcon/symbol.h>
#include <falcon/textwriter.h>

#include <falcon/stringstream.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/statement.h>
#include <falcon/psteps/stmtautoexpr.h>

#include <falcon/parser/parser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/error.h>
#include <falcon/modspace.h>

#include <falcon/expression.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/modloader.h>
#include <falcon/requirement.h>

#include <falcon/errors/genericerror.h>
#include <falcon/errors/syntaxerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/ioerror.h>

#include <falcon/vm.h>

#include <falcon/psteps/exprvalue.h>

#include <falcon/datareader.h>
#include <falcon/importdef.h>
#include <falcon/synclasses_id.h>
#include <falcon/process.h>

namespace Falcon {

//=======================================================================
// context callback
//

IntCompiler::Context::Context(IntCompiler* owner):
   ParserContext( &owner->m_sp ),
   m_owner(owner)
{
}

IntCompiler::Context::~Context()
{
   // nothing to do
}

void IntCompiler::Context::onInputOver()
{
   //std::cout<< "CALLBACK: Input over"<<std::endl;
}


void IntCompiler::Context::onNewFunc( Function* function, Symbol* gs )
{
   SourceParser& sp = m_owner->m_sp;

   if( gs != 0 ) {
      if( ! m_owner->m_mod->addMantraWithSymbol(function, gs, false) ) {
         m_owner->m_sp.addError( e_already_def, sp.currentSource(), sp.currentLine()-1, 0, 0, function->name() );
         delete function;
         return;
      }
   }

   m_owner->m_currentMantra = function;
}


void IntCompiler::Context::onNewClass( Class* cls, bool, Symbol* gs )
{
   SourceParser& sp = m_owner->m_sp;

   if( gs != 0 ) {
      if( ! m_owner->m_mod->addMantraWithSymbol(cls, gs, false) ) {
         m_owner->m_sp.addError( e_already_def, sp.currentSource(), sp.currentLine()-1, 0, 0, cls->name() );
         return;
      }
   }

   FalconClass* fcls = static_cast<FalconClass*>(cls);
   if( fcls->missingParents() == 0 )
   {
      try
      {
         if( ! fcls->construct() )
         {
            m_owner->m_currentMantra = fcls->hyperConstruct();
         }
         else
         {
            m_owner->m_currentMantra = fcls;
         }
      }
      catch( Error* e )
      {
         m_owner->m_sp.addError( e );
         e->decref();
      }
   }
   else
   {
      // we have already raised undefined symbol error.
      // get rid of the class and of all its deps.
      delete cls;
   }
}


void IntCompiler::Context::onNewStatement( Statement* )
{
   // suspend parsing if we're at top level.
}


void IntCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }

   // get the module space
   ModSpace* theSpace = m_owner->m_mod->moduleSpace();

   // do we have a module?
   if( m_owner->m_mod->addLoad( path, isFsPath ) )
   {
      sp.addError( e_load_already, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
      return;
   }

   // just adding it top the module won't be of any help.
   ModLoader* ml = theSpace->modLoader();
   Module* mod = isFsPath ? ml->loadFile( path ) : ml->loadName( path );

   if( mod != 0 )
   {
      theSpace->resolve( mod, true, true ); // will throw on error.
      Error* err = theSpace->link();
      if( err != 0 )
      {
         sp.addError( err );
         err->decref();
      }

      theSpace->readyContext( m_owner->m_vmctx );
   }
   else
   {
      sp.addError( e_mod_notfound, sp.currentSource(), sp.currentLine(), 0, 0, path );
   }
}


bool IntCompiler::Context::onImportFrom( ImportDef* def )
{
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return false;
   }

   // have we already a module group for the module?
   // get the module space
   VMContext* cctx = m_owner->m_vmctx;
   Module* mod = m_owner->m_mod;
   ModSpace* ms = cctx->vm()->modSpace();

   // first, update the module space by pre-loading the required module.
   try
   {
      ms->resolveImportDef( def, 0 );
   }
   catch( Error* e )
   {
      // nope, we can't go on.
      sp.addError(e);
      return false;
   }


   // Now that we know that the module is ok, we can add the import entry to the module.
   Error* linkErrors = mod->addImport( def );

   // if the import was already handled, we should warn he user.
   if( linkErrors != 0 )
   {
      sp.addError( linkErrors );
      return false;
   }

   // the link will resolve newly undefined symbols.
   linkErrors = ms->link();
   if( linkErrors != 0)
   {
      mod->removeImport( def );
      sp.addError(linkErrors);
      return true;
   }

   // When called again, we get the module we just loaded in the ms.
   try
   {
      //... and put it in place.
      ms->resolveImportDef( def, mod );
   }
   catch( Error* e )
   {
      // ... we won't raise errors on second call to resolve, but just in case...
      sp.addError(e);
      return false;
   }

   /** Complete missing imports */
   linkErrors = ms->linkModuleImports( mod );
   if( linkErrors != 0)
   {
      sp.addError(linkErrors);
      return false;
   }

   ms->readyContext( cctx );
   return true;
}


void IntCompiler::Context::onExport(const String& symName )
{
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }

   Module* mod = m_owner->m_mod;

   // do we have a module?
   Symbol* sym = 0;
   bool already;
   sym = mod->addExport( symName, already );

   // already exported?
   if( already )
   {
      sp.addError( e_export_already, m_owner->m_sp.currentSource(),
           sym->declaredAt(), 0, 0 );
      return;
   }

   Error* e = mod->moduleSpace()->exportSymbol( mod, sym );
   if( e != 0 )
   {
      sp.addError( e );
      e->decref();
   }
}


void IntCompiler::Context::onDirective(const String&, const String&)
{
   // TODO
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }
}


void IntCompiler::Context::onGlobal( const String& )
{
   // TODO
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }
}


Symbol* IntCompiler::Context::onUndefinedSymbol( const String& name )
{
   Module* mod = m_owner->m_mod;

   // Is this a global symbol?
   Symbol* gsym = mod->getGlobal( name );
   if (gsym == 0)
   {
      // no? -- create it as an implicit import -- but only if really needed,
      // -- in the interactive mode, we just consider a missing reference to be an error.
      // and try to link it right now.
      Module* importer;
      Symbol* exp = mod->moduleSpace()->findExportedOrGeneralSymbol( mod, name, importer );
      if( exp == 0 )
      {
         return 0;
      }
      gsym = mod->addImplicitImport( name );

      // no need to "define" it, it stays an extern.
      Variable* var = exp->getVariable(0);
      fassert( var != 0 );
      gsym->resolved( var );
   }
   return gsym;
}


Symbol* IntCompiler::Context::onGlobalDefined( const String& name, bool &adef )
{
   Module* mod = m_owner->m_mod;

   Symbol* sym = mod->getGlobal(name);
   if( sym == 0 )
   {
      adef = false;
      Symbol* s = mod->addVariable( name );
      s->declaredAt( m_owner->m_sp.currentLine() );
      return s;
   }

   // The interactive compiler never adds an undefined symbol
   adef = true;
   return sym;
}


Expression* IntCompiler::Context::onStaticData( Class* cls, void* data )
{
   return new ExprValue( Item( cls, data ) );
}


void IntCompiler::Context::onRequirement( Requirement* req )
{
   // the incremental compiler cannot store requirements.
   delete req;
   m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
                req->sourceRef().line(), req->sourceRef().chr(), 0, req->name() );
}


//=======================================================================
// Main class
//

IntCompiler::IntCompiler( bool allowDirective ):
   m_currentTree(0),
   m_bAllowDirective( allowDirective )
{
   // Prepare the compiler and the context.
   m_ctx = new Context( this );
   m_sp.setContext(m_ctx);
   m_sp.interactive(true);
   m_compf = 0;
   m_mod = 0;
   m_vmctx = 0;

   m_lexer = new SourceLexer( "(interactive)", &m_sp );
   m_sp.pushLexer( m_lexer );
}


IntCompiler::~IntCompiler()
{
   delete m_ctx;
   delete m_currentTree;
}


void IntCompiler::setCompilationContext( Function * function, Module* mod, VMContext* vmctx )
{
   m_compf = function;
   m_mod = mod;
   m_vmctx = vmctx;
}


IntCompiler::t_compile_status IntCompiler::compileNext( TextReader* input, SynTree*& code, Mantra*& definition )
{
   m_lexer->setReader(input, false);
   t_compile_status status;

   // create a new syntree if it was emptied
   if( m_currentTree == 0 )
   {
      m_currentTree = new SynTree;
      m_ctx->openMain( m_currentTree );
      m_sp.pushState( "Main", false );
   }

   // if there is a compilation error, throw it
   m_sp.step();

   if( isComplete() )
   {
      if( ! m_currentTree->empty() )
      {
         if ( m_currentTree->at(0)->handler()->userFlags() == FALCON_SYNCLASS_ID_AUTOEXPR )
         {
            // this requires evaluation; but is this a direct call?
            StmtAutoexpr* stmt = static_cast<StmtAutoexpr*>(m_currentTree->at(0));
            Expression* expr = stmt->selector();
            if( expr != 0 && expr->handler()->userFlags() == FALCON_SYNCLASS_ID_CALLFUNC )
            {
               status = e_expr_call;
            }
            else
            {
               status = e_expression;
            }
         }
         else {
            status = e_statement;
         }
         // else ret can stay ok

         m_sp.reset();
         // where to put the tree now?
         // it might be reflected?
         code = m_currentTree;
         m_currentTree = 0;
      }
      else {
         status= e_definition;
         definition = m_currentMantra;
      }
   }
   else {
      status = e_incomplete;
   }

   return status;
}


void IntCompiler::throwCompileErrors() const
{
   class MyEnumerator: public Parsing::Parser::ErrorEnumerator
   {
   public:
      MyEnumerator() {
         myError = new CodeError( e_compile );
      }

      virtual bool operator()( const Parsing::Parser::ErrorDef& def, bool bLast ){

         String sExtra = def.sExtra;
         if( def.nOpenContext != 0 && def.nLine != def.nOpenContext )
         {
            if( sExtra.size() != 0 )
               sExtra += " -- ";
            sExtra += "from line ";
            sExtra.N(def.nOpenContext);
         }

         SyntaxError* err = new SyntaxError( ErrorParam( def.nCode, def.nLine )
               .origin( ErrorParam::e_orig_compiler )
               .chr( def.nChar )
               .module(def.sUri)
               .extra(sExtra));

         myError->appendSubError(err);

         if( bLast )
         {
            throw myError;
         }

         return true;
      }

   private:
      Error* myError;

   } rator;

   m_sp.enumerateErrors( rator );
}

void IntCompiler::resetTree()
{
   m_sp.reset();
   delete m_currentTree;
   delete m_currentMantra;
   m_currentTree = 0;
   m_currentMantra = 0;
}


bool IntCompiler::isComplete() const
{
   return m_sp.isComplete() && m_ctx->isCompleteStatement();
}

}

/* end of intcompiler.h */
