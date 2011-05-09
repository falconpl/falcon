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

#include <falcon/sourcelexer.h>


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

void IntCompiler::Context::onNewFunc( Function* function )
{
   //std::cout<< "CALLBACK: NEW FUNCTION "<< function->name().c_ize() << std::endl;
}


void IntCompiler::Context::onNewClass( Class* cls, bool bIsObj )
{
   //TODO
}


void IntCompiler::Context::onNewStatement( Statement* stmt )
{
   if (currentFunc() == 0)
   {
      m_owner->m_vm->textOut()->write("NEW STATEMENT " + stmt->describe());
   }
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


Symbol* IntCompiler::Context::onGlobalDefined( const String& name )
{
   Symbol* sym = m_owner->m_module->findGlobal(name);
   if( sym == 0 )
   {
      return m_owner->m_module->addVariable( name );
   }
   return sym;
}


void IntCompiler::Context::onUnknownSymbol( UnknownSymbol* sym )
{
   // if we're called here, the symbol is dead and so it's our statement.
   delete sym;
}


void IntCompiler::Context::onStaticData( Class* cls, void* data )
{
 // TODO
}

//=======================================================================
// Main class
//

IntCompiler::IntCompiler( VMachine* vm )
{
   m_vm = vm;

   // Used to keep non-transient data.
   m_module = new Module( "Interactive", false );

   // better for the context to be a pointer, so we can control it's init order.
   Context* m_ctx = new Context( this );
   m_sp.setContext(m_ctx);
   m_sp.interactive(true);

   //TODO: vm->link( m_module );

   // create the streams we're using internally
   m_stream = new StringStream;
   m_reader = new TextReader( m_stream );
   m_writer = new TextWriter( m_stream );

   m_sp.pushLexer(new SourceLexer( "(interactive)", &m_sp, m_reader ) );
}

IntCompiler::~IntCompiler()
{
   // TO be removed:
   delete m_module; // the vm being deleted will kill the module.

   delete m_stream;
   delete m_reader;
   delete m_writer;
   delete m_ctx;
}


IntCompiler::compile_status IntCompiler::compileNext( const String& value )
{
   m_writer->write(value);
   m_writer->flush();
   
   return t_ok;
}


void IntCompiler::resetTree()
{
   m_sp.reset();
}

}

/* end of intcompiler.h */
