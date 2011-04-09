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

#include <falcon/parser/lexer.h>
#include <falcon/textreader.h>

namespace Falcon {
namespace Parser {


Lexer::Lexer( TextReader* reader ):
   m_reader( reader ),
   m_line(0),
   m_chr(0)
{}

Lexer::~Lexer()
{
   delete m_reader;
}

}
}

/* end of parser/lexer.cpp */
