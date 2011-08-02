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


#include <falcon/intcompiler.h>
#include <falcon/module.h>
#include <falcon/symbol.h>
#include <falcon/unknownsymbol.h>
#include <falcon/textwriter.h>
#include <falcon/globalsymbol.h>

#include <falcon/stringstream.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/statement.h>
#include <falcon/stmtautoexpr.h>

#include <falcon/parser/parser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/genericerror.h>
#include <falcon/syntaxerror.h>
#include <falcon/codeerror.h>
#include <falcon/error.h>
#include <falcon/modspace.h>

#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>

#include <falcon/inheritance.h>

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


void IntCompiler::Context::onNewFunc( Function* function, GlobalSymbol* gs )
{
   m_owner->m_module->addFunction( gs, function );
}


void IntCompiler::Context::onNewClass( Class* cls, bool isObject, GlobalSymbol* gs )
{
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   if( fcls->missingParents() == 0 )
   {
      // TODO: Error on failure.
      m_owner->m_module->storeSourceClass( fcls, isObject, gs );        
   }
   else
   {
      // we have already raised undefined symbol error.
      delete cls;
   }
}


void IntCompiler::Context::onNewStatement( Statement* )
{
   // actually nothing to do
}


void IntCompiler::Context::onLoad( const String&, bool )
{
   // TODO
}


void IntCompiler::Context::onImportFrom( const String&, bool, const String&,
         const String&, const String & )
{
   // TODO
}


void IntCompiler::Context::onImport(const String& )
{
   // TODO
}


void IntCompiler::Context::onExport(const String&)
{
   // TODO
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
   // Is this a global symbol?
   Symbol* gsym = m_owner->m_module->getGlobal( name );
   if( gsym == 0 )
   {
      // try to find it in the exported symbols of the VM.
      gsym = (Symbol*) m_owner->m_vm->modSpace()->findExportedSymbol( name );
   }

   return gsym;
}


GlobalSymbol* IntCompiler::Context::onGlobalDefined( const String& name, bool &adef )
{
   Symbol* sym = m_owner->m_module->getGlobal(name);
   if( sym == 0 )
   {
      adef = false;
      return m_owner->m_module->addVariable( name );
   }
   // The interactive compiler never adds an undefined symbol
   // -- as it immediately reports error.
   fassert2( sym->type() == Symbol::t_global_symbol,
         "Undefined symbol in the global map of the interactive compiler." );

   adef = true;
   return static_cast<GlobalSymbol*>(sym);
}


bool IntCompiler::Context::onUnknownSymbol( UnknownSymbol* sym )
{
   // an interactive compiler knows that the target VM is ready.
   // We can dispose of the symbol and signal error returning false.
   delete sym;
   return false;
}


Expression* IntCompiler::Context::onStaticData( Class* cls, void* data )
{
   m_owner->m_module->addStaticData( cls, data );
   return new ExprValue( Item( cls, data ) );
}


void IntCompiler::Context::onInheritance( Inheritance* inh  )
{
   // In the interactive compiler context, classes must have been already defined...
   const Symbol* sym = m_owner->m_module->getGlobal(inh->className());
   if( sym == 0 )
   {
      sym = m_owner->m_vm->modSpace()->findExportedSymbol( inh->className() );
   }

   // found?
   if( sym != 0 )
   {
      Item* itm;
      if( (itm = sym->value( m_owner->m_vm->currentContext() )) == 0 )
      {
         //TODO: Add inheritance line number.
         m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      // we want a class. A real class.
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
      // TODO -- add line
      m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
              inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
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

   // we'll never abandon the main frame in the virtual machine
   m_vm->currentContext()->makeCallFrame( m_main, 0, Item() );

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
         if ( m_currentTree->at(0)->type() == Statement::autoexpr_t )
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

bool IntCompiler::isComplete() const
{
   return m_sp.isComplete() && m_ctx->isCompleteStatement();
}

}

/* end of intcompiler.h */
