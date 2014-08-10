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
#include <falcon/string.h>

namespace Falcon {

class TextReader;

namespace Parsing {

class TokenInstance;
class Parser;

/** Class providing a stream of tokens to be processed one at a time by the parser.

 This is the base class that must be derived in order to provide the parser with
 a stream of tokens that then have to be interpreted.

 The data input is granted by a TextReader that is NOT normally owned by the lexer
 (ownership can be set to a speific base).

 
 */
class FALCON_DYN_CLASS Lexer
{
public:
   /** Creates the lexer.
    * \param uri The URI identifying the source from which this data is read.
    * \param p The parser owning this lexer.
    * \param reader The new text reader.
    *    The input text reader can be specified or changed at a later time.
    *
    */
   Lexer( const String& uri, Parser* p, TextReader* reader = 0 );
   virtual ~Lexer();

   /** Return the next token.
      \return the next token in the stream.
    At stream end, this method returns 0.

   Implementations should return 0 if the current textreader is 0.
    */
   virtual TokenInstance* nextToken() = 0;

   /** Return current lexer line.
      \return current line reached by the lexer.
   */
   int line() const { return m_line; }
   /** Change the current line.
      \param l New current line.

      Useful for macro compilation, or compiling sources in embedded code, so that
      the starting line of the compilation is not 1.
   */
   void line( int l ) { m_line = l; }

   /** Return current lexer character.
      \return current line reached by the lexer.
   */
   int character() const { return m_chr; }

   /** Shortcut to add an error at current line and character. */
   void addError( int code, const String& extra );
   /** Shortcut to add an error at current line and character. */
   void addError( int code );

   const String& uri() const { return m_uri; }
   
   /** Sets the reader (and reader ownership) for this lexer.
    * \param r The new text reader.
    * \param bOwn if true, the reader will be automatically destroyed with this
    *        lexer, or at next setReader() invocation.
    **/
   void setReader( TextReader* r );

protected:
   String m_uri;
   Parser* m_parser;
   TextReader* m_reader;

   int32 m_line;
   int32 m_chr;
   bool m_bReturnOnUnavail;
 };

}
}

#endif	/* _FALCON_PARSER_LEXER_H_ */

/* end of parser/lexer.h */
