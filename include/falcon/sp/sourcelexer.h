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
#include <falcon/textreader.h>

namespace Falcon {

class FALCON_DYN_CLASS SourceLexer: public Parsing::Lexer
{
public:
   SourceLexer( const String& uri, Parsing::Parser* p, TextReader* reader=0 );
   virtual ~SourceLexer();
   
   bool isNameSpace( const String& name );
   void addNameSpace( const String& ns );
   
   virtual Parsing::TokenInstance* nextToken();

   bool isParsingFam() const { return m_bParsingFtd; }
   void setParsingFam( bool m );

private:
   class Private;
   Private* _p;
   
   int32 m_sline;
   int32 m_schr;

   // Used to decide if minus is unary or not.
   bool m_hadOperator;
   bool m_hadImport;

   // Used when starting strings
   bool m_stringStart;

   // true if we allow the string to be multiline.
   bool m_stringML;

   // we're in outscape area.
   bool m_outscape;
   bool m_bParsingFtd;

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
      state_operator,
      state_membuf,
      state_membuf2,
      state_membuf3,
      state_sym_dot
   } t_state;

   t_state m_state;

   typedef enum
   {
      e_st_normal,
      e_st_intl,
      e_st_regex,
      e_st_mutable
   }
   t_string_type;
   t_string_type m_string_type;
   bool m_bRegexIgnoreCase;

   String m_text;

   inline bool isNameCharacter(char_t chr)
   {
      return chr == '_'
             || ( chr >= '0' && chr <= '9')
             || ( chr >= 'A' && chr <= 'Z' )
             || ( chr >= 'a' && chr <= 'z')
             || chr >= 128;
   }

   inline bool isParenthesis(char_t chr)
   {
      return chr == '(' || chr == ')' ||
             chr == '[' || chr == ']' ||
             chr == '{' || chr == '}';
   }

   inline bool isTokenLimit( char_t chr )
   {
      return String::isWhiteSpace( chr )
            || isParenthesis(chr)
            || isTokenStarter(chr)
            || isNameCharacter( chr )
            || chr == '\\'
            ;
   }

   inline bool isTokenStarter( char_t chr )
   {
      return chr == '\'' || chr == '"' || chr == ';' || chr =='^' || chr == ','
          || chr == '@' || chr == '#';
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

   Parsing::TokenInstance* readOutscape();
   Parsing::TokenInstance* makeString();

   bool eatingEOL();
};

}

#endif	/* _FALCON_SOURCELEXER_H */

/* end of sourcelexer.cpp */
