/*
   FALCON - The Falcon Programming Language.
   FILE: parser/lexer.cpp

   Class providing a stream of tokens for the parser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 21:09:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/parser/literal.cpp"

#include <falcon/parser/lexer.h>
#include <falcon/parser/parser.h>
#include <falcon/textreader.h>

namespace Falcon {
namespace Parsing {


Lexer::Lexer( const String& uri, Parser* p, TextReader* reader ):
   m_uri( uri ),
   m_parser(p),
   m_reader( reader ),
   m_line(0),
   m_chr(0)
{
   reader->incref();
}

Lexer::~Lexer()
{
   m_reader->decref();
}

void Lexer::addError( int code, const String& extra )
{
    m_parser->addError( code, m_uri, m_line, m_chr, 0, extra );
}


void Lexer::addError( int code )
{
    m_parser->addError( code, m_uri, m_line, m_chr );
}

void Lexer::setReader( TextReader* reader )
{
   if( reader != 0 )
   {
      reader->incref();
   }
   if (m_reader != 0) {
      m_reader->decref();
   }
   m_reader = reader;
}

}
}

/* end of parser/lexer.cpp */
