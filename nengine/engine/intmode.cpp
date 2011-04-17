/*
   FALCON - The Falcon Programming Language.
   FILE: intmode.cpp

   Interactive mode - step by step dynamic compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/intmode.h>

#include <falcon/vm.h>
#include <falcon/sourceparser.h>
#include <falcon/sourcelexer.h>
#include <falcon/stringstream.h>
#include <falcon/textwriter.h>
#include <falcon/parsercontext.h>

namespace Falcon {

class IntMode::Context: public ParserContext
{
public:
   Context();
   ~Context();

};

//=======================================================================
// Main class
//

IntMode::IntMode()
{
   m_context = new Context(this);
   m_stream = new StringStream;
   m_reader = new TextReader(m_stream);
   m_writer = new TextWriter(m_stream);

   m_vm = new VMachine;
   m_parser = new SourceParser(m_context);
   m_lexer = new SourceLexer(m_reader);

   m_parser->interactive(false);
   m_parser->pushLexer(m_lexer);
   m_parser->pushState("main");
}

IntMode::~IntMode()
{
   delete m_parser;
   delete m_lexer;
   delete m_reader;
   delete m_writer;
   delete m_stream;

   delete m_context;
}


void IntMode::run( const String& snippet )
{
}

}

/* end of intmode.h */
