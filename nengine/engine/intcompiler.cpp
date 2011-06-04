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

#include <falcon/parser/parser.h>
#include <falcon/sourcelexer.h>
#include <falcon/genericerror.h>
#include <falcon/syntaxerror.h>
#include <falcon/codeerror.h>
#include <falcon/error_messages.h>

#include <falcon/expression.h>

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


void IntCompiler::Context::onNewClass( Class* cls, bool bIsObj, GlobalSymbol* gs )
{
   //TODO
}


void IntCompiler::Context::onNewStatement( Statement* stmt )
{
   // actually nothing to do
}


void IntCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   // TODO
}


void IntCompiler::Context::onImportFrom( const String& path, bool isFsPath, const String& symName,
         const String& asName, const String &inName )
{
   // TODO
}


void IntCompiler::Context::onImport(const String& symName )
{
   // TODO
}


void IntCompiler::Context::onExport(const String& symName)
{
   // TODO
}


void IntCompiler::Context::onDirective(const String& name, const String& value)
{
   // TODO
}


void IntCompiler::Context::onGlobal( const String& name )
{
   // TODO
}


Symbol* IntCompiler::Context::onUndefinedSymbol( const String& name )
{
   // Is this a global symbol?
   Symbol* gsym = m_owner->m_module->findGlobal( name );
   if( gsym == 0 )
   {
      // try to find it in the exported symbols of the VM.
      //TODO
      // gsym = m_owner->m_vm->findPublicSymbol( sym->name() )
   }

   return gsym;
}


GlobalSymbol* IntCompiler::Context::onGlobalDefined( const String& name, bool &adef )
{
   GlobalSymbol* sym = m_owner->m_module->findGlobal(name);
   if( sym == 0 )
   {
      adef = false;
      return m_owner->m_module->addVariable( name );
   }
   adef = true;
   return sym;
}


void IntCompiler::Context::onUnknownSymbol( UnknownSymbol* sym )
{
   // there's nothing useful we may want to do here (at least for now)
}


void IntCompiler::Context::onStaticData( Class* cls, void* data )
{
 // TODO
}

//=======================================================================
// Main class
//

IntCompiler::IntCompiler( VMachine* vm ):
   m_currentTree(0)
{
   // prepare the virtual machine.
   m_vm = vm;
   // Used to keep non-transient data.
   m_module = new Module( "Interactive", false );
   m_main = new SynFunc("__main__" );
   // as the module is dynamic, m_main will be destroyed by m_module
   m_module->addFunction(m_main, false);

   // we'll never abandon the main frame in the virtual machine
   m_vm->currentContext()->makeCallFrame( m_main, 0, Item(), false );

   // Link the module so that the VM knows about it (and protects its global).
   //TODO: vm->link( m_module );

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
   // TO be removed:
   delete m_module; // the vm being deleted will kill the module.

   delete m_stream;
   delete m_writer;
   delete m_ctx;

   delete m_currentTree;
}


IntCompiler::compile_status IntCompiler::compileNext( const String& value )
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
      m_sp.pushState( "Main" );
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
         m_vm->pushReturn();
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

         delete m_currentTree;
         m_currentTree = 0;
      }

      return ret;
   }

   return incomplete_t;
}


void IntCompiler::throwCompileErrors() const
{
   class MyEnumerator: public Parsing::Parser::errorEnumerator
   {
   public:
      MyEnumerator() {
         myError = new CodeError( e_compile );
      }

      virtual bool operator()( const Parsing::Parser::ErrorDef& def, bool bLast ){

         String sExtra = def.sExtra;
         if( def.nOpenContext != 0 )
         {
            if( sExtra.size() != 0 )
               sExtra += " -- ";
            sExtra += "from line ";
            sExtra.N(def.nOpenContext);
         }

         SyntaxError* err = new SyntaxError( ErrorParam( def.nCode, def.nLine )
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
