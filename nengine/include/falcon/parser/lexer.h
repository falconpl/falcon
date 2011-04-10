/*
   FALCON - The Falcon Programming Language.
   FILE: parser/lexer.h

   Class providing a stream of tokens for the parser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 21:09:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_LEXER_H_
#define	_FALCON_PARSER_LEXER_H_

#include <falcon/setup.h>

#include <falcon/parser/tint.h>
#include <falcon/parser/tfloat.h>
#include <falcon/parser/tstring.h>
#include <falcon/parser/tname.h>
#include <falcon/parser/teol.h>
#include <falcon/parser/teof.h>


namespace Falcon {

class TextReader;

namespace Parser {

class TokenInstance;
class Parser;

/** Class providing a stream of tokens to be processed one at a time by the parser.

 This is the base class that must be derived in order to provide the parser with
 a stream of tokens that then have to be interpreted.

 The data input is granted by a TextReader that is owned by the lexer (destroyed
 at lexer destruction).
 
 */
class FALCON_DYN_CLASS Lexer
{
public:
   
   Lexer( const String& uri, Parser* p, TextReader* reader );
   virtual ~Lexer();

   /** Return the next token.
      \return the next token in the stream.
    At stream end, this method returns 0.

    */
   virtual TokenInstance* nextToken() = 0;

   /** Return current lexer line.
      \return current line reached by the lexer.
   */
   int line() const { return m_line; }

   /** Return current lexer character.
      \return current line reached by the lexer.
   */
   int character() const { return m_chr; }

   /** Shortcut to add an error at current line and character. */
   void addError( int code, const String& extra );
   /** Shortcut to add an error at current line and character. */
   void addError( int code );

protected:
   String m_uri;
   Parser* m_parser;
   TextReader* m_reader;

   int32 m_line;
   int32 m_chr;
 };

}
}

#endif	/* _FALCON_PARSER_LEXER_H_ */

/* end of parser/lexer.h */
