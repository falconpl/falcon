/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_lexer.h

   Lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_LEXER_H
#define FALCON_LEXER_H

#include <falcon/setup.h>
#include <falcon/stream.h>
#include <falcon/string.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

namespace Falcon
{

class SyntreeElement;
class Compiler;
class Stream;

class FALCON_DYN_CLASS SrcLexer: public BaseAlloc
{
public:
   /** Type of context */
   typedef enum {
      /** Topmost context */
      ct_top,
      /** Inside round parenthesis */
      ct_round,
      /** Inside square parenthesis */
      ct_square,
      /** Inside a string (of any kind; the kind is determined by the status) */
      ct_string,
      /** Inside function declaration */
      ct_graph,
      /** Inside an inner function */
      ct_inner
   } t_contextType;
   
private:

   
   class Context: public BaseAlloc
   {
   public:
      t_contextType m_ct;
      int m_oline;
      Context* m_prev;
      
      Context( t_contextType ct, int openLine, Context* prev=0):
         m_ct( ct ),
         m_oline( openLine ),
         m_prev( prev )
      {
      }
   };
    
   void *m_value;
   int m_line;
   int m_previousLine;
   int m_character;
   int m_prevStat;
   bool m_firstEq;
   bool m_done;
   bool m_addEol;
   bool m_lineFilled;
   bool m_bIsDirectiveLine;
   bool m_incremental;
   bool m_lineContContext;
   bool m_graphAgain;
   uint32 m_chrEndString;
   bool m_mlString;

   Stream *m_in;
   List m_streams;
   List m_streamLines;

   Compiler *m_compiler;
   String m_string;

   typedef enum
   {
      e_line,
      e_string,
      e_stringOctal,
      e_stringBin,
      e_stringHex,
      e_stringRunning,
      e_loadDirective,
      e_eolComment,
      e_blockComment,
      e_zeroNumber,
      e_intNumber,
      e_binNumber,
      e_octNumber,
      e_hexNumber,
      e_floatNumber,
      e_floatNumber_e,
      e_floatNumber_e1,
      e_operator,
      e_symbol,
      e_litString
   } t_state;

   t_state m_state;

   typedef enum
   {
      t_mNormal,
      t_mOutscape,
      t_mEval
   } t_mode;

   t_mode m_mode;
   bool m_bParsingFtd;
   bool m_bWasntEmpty;
   String m_whiteLead;

   Context* m_topCtx;
   //==================

   /** Scans m_string for recognized tokens, and eventually returns them. */
   int checkUnlimitedTokens( uint32 nextChar );

   int checkLimitedTokens();
   void checkContexts();

   bool isWhiteSpace( uint32 chr ) {
      return chr == ' ' || chr == '\t' || chr == '\r' || chr == '\b' || chr == 0x12
         || chr == 0x3000 || chr == 0x00A0;  // unicode ideographic & nbs
   }

   /** Special (non symbolic) high characters. */
   bool isSpecialChar( uint32 chr )
   {
      // special spaces...
      return chr == 0x3000 || chr == 0x00A0 ||
            // Special quotes
            chr == 0x201C || chr == 0x201D ||
            chr == 0x300C || chr == 0x300D ||
            chr == 0xFF09 || chr == 0xFF08;
   }

   bool isSymbolChar( uint32 chr )
   {
      return (chr >= '0' && chr <= '9') ||
                 (chr >= 'a' && chr <= 'z') ||
                 (chr >= 'A' && chr <= 'Z') ||
                 chr == '_' ||
                 ( chr > 0x80 && ! isSpecialChar(chr) );
   }

   bool isTokenLimit( uint32 chr )
   {
      return ! isSymbolChar( chr );
   }

   // Process the state line
   int state_line( uint32 chr );

   int lex_normal();
   int lex_outscape();
   int lex_eval();

public:
   SrcLexer( Compiler *comp );
   ~SrcLexer();

   /** Return current line in file (starting from 1). */
   int line() const {
      return m_line;
   }

   /** Return current character position in current line (starting from 1). */
   int character() const {
      return m_character;
   }

   int previousLine() const {
      return m_previousLine;
   }

   void resetContexts();

   void line( int val ) { m_line = val; }

   /** Hook for bison.
      Bison will call this with his "parameter", which is the value
      of the various expression.
      The type for lexed values is the lex_value_t union, declared
      in the bison files.
   */
   int doLex( void *param )
   {
      m_value = param;
      return lex();
	}

   void *value() const { return m_value; }
   int lex();

   void input( Stream *i );
   Stream *input() const { return m_in; }

   /** Resets the lexer, preparing it for another compilation. */
   void reset();

   bool parsingFtd() const { return m_bParsingFtd; }
   void parsingFtd( bool b );

   bool hasOpenContexts() { return m_topCtx != 0 || m_lineContContext; }
   bool incremental() const { return m_incremental; }
   void incremental( bool b ) { m_incremental = b; }

   void appendStream( Stream *s );
   void parseMacro();
   void parseMacroCall();

   /** Add a compilation lexer context. */
   void pushContext( t_contextType ct, int startLine );
   
   /** Pops a compilation lexer context. 
     \return true if the contest was popped, false if we had no context.
   */
   bool popContext();
   
   /** Reads the current context type.
      \return ct_top if outside any context.
   */
   t_contextType currentContext();
   
   /** Gets the line at which the current context started. 
   \return 0 if no context is open, a valid line otherwise.
   */
   int contextStart();
   
   /** Determines if the current context is a parentetic context.
      \return true if in parenthesis.
   */
   bool inParCtx();

   /** Specialized version of ReadAhead, killing unneeded \\r
      \return true if the character could be read ahead
      \param chr Where to place the read character
   */
   bool readAhead( uint32 &chr );
};

}

#endif

/* end of falcon_lexer.h */
