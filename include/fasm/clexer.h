/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_lexer.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FASM_CLEXER_H
#define FASM_CLEXER_H

#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/basealloc.h>

namespace Falcon
{

class Module;
class AsmCompiler;

class FALCON_DYN_CLASS AsmLexer: public BaseAlloc
{
private:
   Pseudo **m_value;
   int m_line;
   int m_character;
   String m_string;
   int m_prev_stat;

   Pseudo m_rega;
   Pseudo m_regb;
   Pseudo m_regs1;
   Pseudo m_regs2;
   Pseudo m_regs3;
   Pseudo m_nil;
   Pseudo m_true;
   Pseudo m_false;

   Module *m_module;
   AsmCompiler *m_compiler;
   Stream *m_in;
   bool m_done;

   typedef enum
   {
      e_line,
      e_string,
      e_stringOctal,
      e_stringHex,
      e_stringID,
      e_directive,
      e_comment,
      e_zeroNumber,
      e_intNumber,
      e_octNumber,
      e_hexNumber,
      e_floatNumber,
      e_floatNumber_e,
      e_floatNumber_e1,
      e_word
   } t_state;

   t_state m_state;

   int checkTokens();

   // Process the state line
   int state_line( uint32 chr );

   bool isWhiteSpace( uint32 chr ) {
      return chr == ' ' || chr == '\t' || chr == '\r' || chr == '\b' || chr == 0x12;
   }

   bool isTokenChar( uint32 chr )
   {
      return (chr >= '0' && chr <= '9') ||
                 (chr >= 'a' && chr <= 'z') ||
                 (chr >= 'A' && chr <= 'Z') ||
                 chr == '_' || chr == '#' || chr == '*' || chr == '.' || chr == '%' ||
                 chr >= 0x80;
   }

   bool isTokenLimit( uint32 chr )
   {
      return ! isTokenChar( chr );
   }

   int checkDirectives();
   int checkPostDirectiveTokens();
   bool m_bDirective;

public:
   AsmLexer( Module *mod, AsmCompiler *cmp, Stream *in );

   int line() const { return m_line; }
   int character() const { return m_character; }

   /** Hook for bison.
      Bison will call this with his "parameter", which is the value
      of the various expression. Bison type for falcon assembly is
      Falcon::Pseudo *, no mistake in casting.
   */
   int doLex( void *param )
   {
      m_value = static_cast<Pseudo **>( param );
      return lex();
   }

   Pseudo **value() const { return m_value; }

   /** Lex routine */
   int lex();

   Pseudo *regA_Inst() { return &m_rega; }
   Pseudo *regB_Inst() { return &m_regb; }
   Pseudo *regS1_Inst() { return &m_regs1; }
   Pseudo *regS2_Inst() { return &m_regs2; }
   Pseudo *true_Inst() { return &m_true; }
   Pseudo *false_Inst() { return &m_false; }
   Pseudo *nil_Inst() { return &m_nil; }
};


}

#endif

/* end of falcon_lexer.h */
