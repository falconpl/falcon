/*
 * testparser.cpp
 *
 *  Created on: 15/gen/2011
 *      Author: gian
 */

#include <cstdio>
#include <iostream>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/error.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>
#include <falcon/statement.h>
#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>
#include <falcon/inheritance.h>
#include <falcon/requirement.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>

#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parsercontext.h>

#include <falcon/textreader.h>
#include <falcon/globalsymbol.h>
#include <falcon/genericerror.h>
#include <falcon/unknownsymbol.h>

#include <falcon/falconclass.h>

using namespace Falcon;

//==============================================================
// The compiler context
//
class Context: public ParserContext
{
public:
   Context(SourceParser* parser);
   ~Context();
   void display();

   virtual void onInputOver();
   virtual void onNewFunc( Function* function, GlobalSymbol* gs = 0 );
   virtual void onNewClass( Class* cls, bool bIsObj, GlobalSymbol* gs = 0 );
   virtual void onNewStatement( Statement* stmt );
   virtual void onLoad( const String& path, bool isFsPath );
   virtual void onImportFrom( const String& path, bool isFsPath, const String& symName,
         const String& nsName, bool bIsNS );
   virtual void onImport(const String& symName );
   virtual void onExport(const String& symName);
   virtual void onDirective(const String& name, const String& value);
   virtual void onGlobal( const String& name );
   virtual Symbol* onUndefinedSymbol( const String& name );
   virtual GlobalSymbol* onGlobalDefined( const String& name, bool& bUnique );
   virtual bool onUnknownSymbol( UnknownSymbol* sym );
   virtual Expression* onStaticData( Class* cls, void* data );
   virtual void onInheritance( Inheritance* );
   virtual void onRequirement( Requirement* rec );

private:
   SynFunc m_main;
   StdInStream m_stdin;
   TextReader* m_input;

};

Context::Context(SourceParser* parser):
   ParserContext( parser ),
   m_main("__main__"),
   m_stdin(false)

{
   m_input = new TextReader(&m_stdin);
   SourceLexer* lexer = new SourceLexer("stdin", parser, m_input);
   parser->pushLexer(lexer); // the parser will dispose of the lexer when needed
   openMain( &m_main.syntree() ); 
}

Context::~Context()
{
   // nothing to do
}

void Context::display()
{
   std::cout << "Parsed code: "<<
      std::endl << m_main.syntree().describe().c_ize() << std::endl;
}

void Context::onInputOver()
{
   std::cout<< "CALLBACK: Input over"<<std::endl;
}

void Context::onNewFunc( Function* function, GlobalSymbol*)
{
   std::cout<< "CALLBACK: NEW FUNCTION "<< function->name().c_ize() << std::endl;
}


void Context::onNewClass( Class* cls, bool bIsObj, GlobalSymbol* )
{
   std::cout<< "CALLBACK: New class "<< cls->name().c_ize()
      << (bIsObj ? " (object)":"") << std::endl;
}

void Context::onNewStatement( Statement* stmt )
{
   std::cout<< "CALLBACK: New statement "<< stmt->oneLiner().c_ize() << std::endl;
}

void Context::onLoad( const String& path, bool isFsPath )
{
   std::cout<< "CALLBACK: Load "<< path.c_ize() << (isFsPath ? " (path)" : "") << std::endl;
}

void Context::onImportFrom( const String& path, bool isFsPath, const String& symName,
         const String& nsName, bool bIsNS )
{
   std::cout << "CALLBACK: import " << symName.c_ize() << " from "
      << path.c_ize() << (isFsPath ? " (path) " : " ")
      << (nsName.size() ? (String( bIsNS ? " in ": " as ")+nsName).c_ize() : "")
      << std::endl;
}

void Context::onImport(const String& symName )
{
   std::cout << "CALLBACK: import " << symName.c_ize() << std::endl;
}


void Context::onExport(const String& symName)
{
   std::cout << "CALLBACK: export " << symName.c_ize() << std::endl;
}

void Context::onDirective(const String& name, const String& value)
{
   std::cout << "CALLBACK: directive " << name.c_ize() << " = " << value.c_ize() << std::endl;
}


void Context::onGlobal( const String& name )
{
   std::cout << "CALLBACK: global " << name.c_ize() << std::endl;
}


Symbol* Context::onUndefinedSymbol( const String& name )
{
   std::cout << "CALLBACK: undefined " << name.c_ize() << std::endl;
   return m_main.symbols().addLocal(name);
}

GlobalSymbol* Context::onGlobalDefined( const String& name, bool& )
{
   std::cout << "CALLBACK: new global defined: " << name.c_ize() << std::endl;
   GlobalSymbol* sym = new GlobalSymbol(name,0);
   m_main.symbols().addSymbol(sym);
   return sym;
}

bool Context::onUnknownSymbol( UnknownSymbol* sym )
{
   std::cout << "CALLBACK: unknwon symbol -- pretty impossible: " <<
         sym->name().c_ize() << std::endl;
   delete sym;
   return false;
}

 Expression* Context::onStaticData( Class* cls, void* data )
 {
    String temp;
    cls->describe( data, temp );

    std::cout << "CALLBACK: static data : " <<
         temp.c_ize() << std::endl;

    return new ExprValue( Item( cls, data, true ) );
 }


 void Context::onInheritance( Inheritance* inh )
 {
    String temp;
    inh->describe( temp );

    std::cout << "CALLBACK: inheritance data : " <<
         temp.c_ize() << std::endl;
    // for now, we'll let it leak.
}

 void Context::onRequirement( Requirement* req )
 {
    String temp = req->name();

    std::cout << "CALLBACK: requirement data : " <<
         temp.c_ize() << std::endl;
    // for now, we'll let it leak.
 }
  
//==============================================================
// The application
//

class ParserApp: public Falcon::Application
{

public:
   void guardAndGo()
   {
      try {
         go();
      }
      catch( Error* e )
      {
         std::cout << "Caught: " << e->describe().c_ize() << std::endl;
         e->decref();
      }
   }

void go()
{

   SourceParser parser;
   Context myContext(&parser);

   if( parser.parse() )
   {
      myContext.display();
   }
   else
   {
      throw parser.makeError();
   }
}

};

// This is just a test.
int main( int , char* [] )
{
   std::cout << "Parser test!" << std::endl;

   TRACE_ON();

   ParserApp app;
   app.guardAndGo();

   return 0;
}

