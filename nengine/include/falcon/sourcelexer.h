/*
   FALCON - The Falcon Programming Language
   FILE: sourcelexer.h

   Lexer detecting the tokens for Falcon script sources
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 18:13:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SOURCELEXER_H
#define	_FALCON_SOURCELEXER_H

#include <falcon/parser/lexer.h>

#include "parser/tokeninstance.h"
#include "textreader.h"

namespace Falcon {

class FALCON_DYN_CLASS SourceLexer: public Parsing::Lexer
{
public:
   SourceLexer( const String& uri, Parsing::Parser* p, TextReader* reader );
   virtual ~SourceLexer();
   
   virtual Parsing::TokenInstance* nextToken();

private:
   int32 m_sline;
   int32 m_schr;

   // Used to decide if minus is unary or not.
   bool m_hadOperator;

   // Used when starting strings
   bool m_stringStart;

   // true if we allow the string to be multiline.
   bool m_stringML;

   typedef enum
   {
      state_none,
      state_line,
      state_shebang1,
      state_shebang2,


      state_enterComment,
      state_eolComment,
      state_blockComment,
      state_blockCommentAsterisk,
      state_blockCommentEscape,

      state_double_string,
      state_double_string_nl,
      state_double_string_esc,

      state_double_string_hex,
      state_double_string_bin,
      state_double_string_octal,
              
      state_single_string,

      state_zero_prefix,
      state_integer,
      state_octal,
      state_hex,
      state_bin,
      state_float,   
      state_float_first,
      state_float_exp,

      state_name,
      state_operator
   } t_state;

   t_state m_state;

   String m_text;
   Parsing::TokenInstance* m_nextToken;

   inline bool isTokenLimit(char_t chr)
   {
      return chr != '_' &&
             (  chr < '0' ||
                (chr > '9' && chr < 'A' ) ||
                (chr > 'Z' && chr < 'a' ) ||
                (chr > 'z' && chr < 128 ) );
   }

   inline bool isParenthesis(char_t chr)
   {
      return chr == '(' || chr == ')' ||
             chr == '[' || chr == ']' ||
             chr == '{' || chr == '}';
   }

   inline bool isCipher(char_t chr)
   {
      return chr >= '0' && chr <= '9';
   }

   inline bool isLetter(char_t chr)
   {
      return (chr >= 'A' && chr <= 'Z') || (chr >= 'a' && chr <= 'z');
   }

   inline bool isSymStart(char_t chr)
   {
      return isLetter(chr) || chr >= 128 || chr == '_';
   }

   inline void unget( char_t chr )
   {
      m_reader->ungetChar(chr);
   }

   Parsing::TokenInstance* checkOperator();
   Parsing::TokenInstance* checkWord();
   void resetState();
};

}

#endif	/* _FALCON_SOURCELEXER_H */

/* end of sourcelexer.cpp */
