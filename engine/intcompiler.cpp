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
#include <falcon/inheritance.h>
#include <falcon/requirement.h>

#include <falcon/errors/genericerror.h>
#include <falcon/errors/syntaxerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/ioerror.h>

#include <falcon/vm.h>

#include <falcon/psteps/exprvalue.h>

#include "falcon/datareader.h"
#include "falcon/importdef.h"


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
   try {
      // remove the function from the static-data
      if( gs != 0 )
      {
         m_owner->m_module->addFunction( gs, function );
      }
      else
      {
         m_owner->m_module->addAnonFunction( function );
      }
   }
   catch( Error* e )
   {
      m_owner->m_sp.addError( e );
      e->decref();
   }
}


void IntCompiler::Context::onNewClass( Class* cls, bool isObject, Symbol* gs )
{
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   if( fcls->missingParents() == 0 )
   {
      // TODO: Error on failure (?)
      try
      {
         m_owner->m_module->storeSourceClass( fcls, isObject, gs );        
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
   // actually nothing to do
}


void IntCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->m_sp;
   if( ! m_owner->m_module->addLoad( path, isFsPath ) )
   {
      sp.addError( e_load_already, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
   }
   else
   {
      // just adding it top the module won't be of any help.
      ModLoader* ml = m_owner->m_vm->modLoader();
      Module* mod = isFsPath ? ml->loadFile( path ) : ml->loadName( path );
      
      if( mod != 0 )
      {
         ModSpace* theSpace = m_owner->m_vm->modSpace();
         theSpace->resolve( ml, mod, true, true ); // will throw on error.
         Error* err = theSpace->link();
         if( err != 0 )
         {
            sp.addError( err );
            err->decref();
         }
         
         theSpace->readyVM( m_owner->m_vm->currentContext() );
      }
      else
      {
         sp.addError( e_mod_notfound, sp.currentSource(), sp.currentLine(), 0, 0, path );
      }
   }
}


bool IntCompiler::Context::onImportFrom( ImportDef* def )
{
   SourceParser& sp = m_owner->m_sp;

   // have we already a module group for the module?
   Module* vmmod = m_owner->m_module;
   ModSpace* ms = vmmod->moduleSpace(); 
   
   // first, update the module space by pre-loading the required module.
   try
   {
      ms->resolveImportDef( def, m_owner->m_vm->modLoader(), 0 );
   }
   catch( Error* e )
   {
      // nope, we can't go on.
      sp.addError(e);
      return false;
   }
   
   // Now that we know that the module is ok, we can add the import entry to the module.
   Error* linkErrors = vmmod->addImport( def );
   
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
      vmmod->removeImport( def );
      sp.addError(linkErrors);
      return true;
   }
   
   // When called again, we get the module we just loaded in the ms.    
   try
   {
      //... and put it in place.
      ms->resolveImportDef( def, m_owner->m_vm->modLoader(), vmmod );
   }
   catch( Error* e )
   {
      // ... we won't raise errors on second call to resolve, but just in case...
      sp.addError(e);
      return false;
   }

   /** Complete missing imports */
   linkErrors = ms->linkModuleImports( vmmod );
   if( linkErrors != 0) 
   {
      sp.addError(linkErrors);
      return false;
   }
   
   ms->readyVM( m_owner->m_vm->currentContext() );
   return true;
}


void IntCompiler::Context::onExport(const String& symName )
{
   bool already;
   Symbol* sym = m_owner->m_module->addExport( symName, already );
   
   // already exported?
   if( already )
   {
      m_owner->m_sp.addError( e_export_already, m_owner->m_sp.currentSource(), 
           sym->declaredAt(), 0, 0 );
   }
   else
   {
      
      Error* e = m_owner->m_vm->modSpace()->exportSymbol( m_owner->m_module, sym );
      if( e != 0 )
      {
         SourceParser& sp = m_owner->m_sp;
         sp.addError( e );
         e->decref();
      }
   }
}


void IntCompiler::Context::onDirective(const String&, const String&)
{
   // TODO
}


void IntCompiler::Context::onGlobal( const String& )
{
   // TODO
}


Symbol* IntCompiler::Context::onUndefinedSymbol( const String& name )
{
   Module* mod = m_owner->m_module;
   // Is this a global symbol?
   Symbol* gsym = mod->getGlobal( name );
   if (gsym == 0)
   {
      // no? -- create it as an implicit import -- but only if really needed,
      // -- in the interactive mode, we just consider a missing reference to be an error.
      // and try to link it right now.
      Symbol* exp = mod->moduleSpace()->findExportedOrGeneralSymbol( mod, name );
      if( exp == 0 )
      {
         return 0;
      }
      gsym = mod->addImplicitImport( name );  
      
      // no need to "define" it, it stays an extern.
      gsym->id( exp->id() );
      gsym->defaultValue( exp->defaultValue() );
   }
   return gsym;
}


Symbol* IntCompiler::Context::onGlobalDefined( const String& name, bool &adef )
{
   Symbol* sym = m_owner->m_module->getGlobal(name);
   if( sym == 0 )
   {
      adef = false;
      return m_owner->m_module->addVariable( name );
   }
   // The interactive compiler never adds an undefined symbol
   // -- as it immediately reports error.
   fassert2( sym->type() != Symbol::e_st_undefined,
         "Undefined symbol in the global map of the interactive compiler." );

   adef = true;
   return sym;
}


Expression* IntCompiler::Context::onStaticData( Class* cls, void* data )
{
   m_owner->m_module->addStaticData( cls, data );
   if( cls->typeID() == FLC_ITEM_FUNC )
   {
      // simplify functions to function items.
      return new ExprValue( Item( static_cast<Function*>(data) ) );
   }
   else
   {
      return new ExprValue( Item( cls, data ) );
   }
}


void IntCompiler::Context::onInheritance( Inheritance* inh )
{
   Module* mod = m_owner->m_module;
   // In the interactive compiler context, classes must have been already defined...
   const Symbol* sym = mod->getGlobal(inh->className());
  
   // found?
   if( sym != 0 )
   {
      Item* itm = sym->defaultValue();
      if ( ! itm->isUser() || ! itm->asClass()->isMetaClass() )
      {
         m_owner->m_sp.addError( e_inv_inherit, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      else
      {
         inh->parent( static_cast<Class*>(itm->asInst()) );
      }
   }
   else
   {      
      m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
   }  
}


void IntCompiler::Context::onRequirement( Requirement* rec )
{
   // In the interactive compiler context, classes must have been already defined...
   try
   {
      m_owner->m_module->addRequirement( rec );  
   }
   catch( Error *e )
   {
      m_owner->m_sp.addError( e );
      e->decref();
   }
}

//=======================================================================
// Main class
//

IntCompiler::IntCompiler( VMachine* vm ):
   m_currentTree(0)
{
   // prepare the virtual machine.
   m_vm = vm;

   m_module = new Module( "Interactive" );
   m_main = new SynFunc("__main__" );
   m_module->addFunction(m_main, false);
   
   m_vm->modSpace()->add(m_module, true, false);
   
   // we'll never abandon the main frame in the virtual machine
   m_vm->currentContext()->makeCallFrame( m_main, 0, Item() );
   // and we know the code up to here is safe.
   m_vm->currentContext()->setSafeCode();

   // Prepare the compiler and the context.
   m_ctx = new Context( this );
   m_sp.setContext(m_ctx);
   m_sp.interactive(true);

   // create the streams we're using internally
   m_stream = new StringStream;
   m_writer = new TextWriter( m_stream );

   m_sp.pushLexer(new SourceLexer( "(interactive)", &m_sp, new TextReader( m_stream ) ) );
}


IntCompiler::~IntCompiler()
{
   delete m_module; 

   delete m_writer;
   delete m_stream;
   delete m_ctx;

   delete m_currentTree;
}


IntCompiler::compile_status IntCompiler::compileNext( const String& value)
{
   // write the new data on the stream.
   int64 current = m_stream->seekCurrent(0);
   m_writer->write(value);
   m_writer->flush();
   m_stream->seekBegin(current);

   // create a new syntree if it was emptied
   if( m_currentTree == 0 )
   {
      m_currentTree = new SynTree;
      m_ctx->openMain( m_currentTree );
      m_sp.pushState( "Main", false );
   }

   // if there is a compilation error, throw it
   if( ! m_sp.step() )
   {
      throwCompileErrors();
   }

   if( isComplete() )
   {
      compile_status ret = ok_t;
      if( ! m_currentTree->empty() )
      {
         VMContext* ctx = m_vm->currentContext();
         ctx->pushReturn();
         ctx->pushCode(m_currentTree);
         if ( m_currentTree->at(0)->type() == Statement::e_stmt_autoexpr )
         {
            // this requires evaluation; but is this a direct call?
            StmtAutoexpr* stmt = static_cast<StmtAutoexpr*>(m_currentTree->at(0));
            Expression* expr = stmt->expr();
            if( expr != 0 && expr->type() == Expression::t_funcall )
            {
               ret = eval_direct_t;
            }
            else
            {
               ret = eval_t;
            }
         }
         // else ret can stay ok

         try {
            m_vm->run();
         }
         catch(...)
         {
            m_sp.reset();
            delete m_currentTree;
            m_currentTree = 0;
            throw;
         }

         m_sp.reset();
         delete m_currentTree;
         m_currentTree = 0;
      }

      return ret;
   }

   return incomplete_t;
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
}


void IntCompiler::resetVM()
{
   VMContext* ctx = m_vm->currentContext();
   ctx->reset();
   ctx->makeCallFrame( m_main, 0, Item() );
}


bool IntCompiler::isComplete() const
{
   return m_sp.isComplete() && m_ctx->isCompleteStatement();
}

}

/* end of intcompiler.h */