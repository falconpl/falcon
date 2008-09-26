/*
   FALCON - The Falcon Programming Language
   FILE: fasm_lexer.cpp

   Assembly lexer implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 27 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Assembly lexer implementation.
*/

#include <fasm/comp.h>
#include <fasm/clexer.h>
#include <falcon/error.h>
#include <fasm/pseudo.h>
#include <falcon/module.h>
#include "fasm_parser.h"

#define VALUE  (value())

namespace Falcon {


AsmLexer::AsmLexer( Module *mod, AsmCompiler *cmp, Stream *in ):
      m_value(0),
      m_line( 1 ),
      m_character( 0 ),
      m_prev_stat(0),
      
      m_module( mod ),
      m_compiler( cmp ),   
      m_in( in ),
      m_done( false ),
      m_bDirective( true ),
      
      m_rega( Pseudo::tregA, false),
      m_regb( Pseudo::tregB, false ),
      m_regs1( Pseudo::tregS1, false ),
      m_regs2( Pseudo::tregS2, false ),
   	m_regl1( Pseudo::tregL1, false ),
      m_regl2( Pseudo::tregL2, false ),
      m_nil( Pseudo::tnil, false ),
      m_true( Pseudo::imm_true, false ),
      m_false( Pseudo::imm_false, false ),
      
      m_state( e_line )
{}

int AsmLexer::lex()
{
   if ( m_done )
      return 0;

   // reset previous token
   m_string.size(0);
   String tempString;
   uint32 chr;

   bool next_loop = true;

   while( next_loop )
   {
      next_loop = m_in->get( chr );

      if ( ! next_loop )
      {
         // fake an empty terminator at the end of input.
         m_done = true;
         chr = '\n';
      }

      m_character++;


      switch ( m_state )
      {
         case e_line:
            // in none status, we have to discard blanks and even ignore '\n';
            // we enter in line or string status depending on what we find.
            if( ! isWhiteSpace( chr ) )
            {
               // whitespaces and '\n' can't follow a valid symbol,
               // as since they are token limiters, and they would be read
               // ahead after token begin, valid symbols and token has already
               // been returned.

               int token = state_line( chr );
               if ( token != 0 )
                  return token;
            }
         break;

         case e_comment:
            if ( chr == '\n' )
            {
               m_line ++;
               m_character = 0;
               m_state = e_line;
               m_bDirective = false;
               return EOL;
            }
         break;

         case e_stringID:
            if ( chr < '0' || chr > '9' )
            {
               // end
               m_in->unget( chr );
               int64 retval;
               if ( ! m_string.parseInt( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               const Falcon::String *str = m_module->getString( (uint32) retval );
               m_state = e_line;
               if ( str == 0 ) {
                  m_compiler->raiseError( e_str_noid );
               }
               else {
                  *VALUE = new Falcon::Pseudo( m_line, Falcon::Pseudo::imm_string, retval );
                  return STRING_ID;
               }
            }
         break;

         case e_intNumber:
            if ( chr == '.' )
            {
               m_string.append( chr );
               m_state = e_floatNumber;
            }
            else if ( chr == 'e' )
            {
               m_string.append( chr );
               m_state = e_floatNumber_e;
            }
            else if ( chr < '0' || chr > '9' )
            {
               // end
               m_in->unget( chr );
               int64 retval;
               if ( ! m_string.parseInt( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               *VALUE = new Falcon::Pseudo( line(), retval );
               m_state = e_line;
               return INTEGER;
            }
            else
               m_string.append( chr );
         break;

         case e_floatNumber:
            if ( chr == 'e' )
            {
               m_state = e_floatNumber_e;
               m_string.append( chr );
            }
            else if ( (chr < '0' || chr > '9') && chr != '.' )
            {
               // end
               m_in->unget( chr );
               numeric retval;
               if ( ! m_string.parseDouble( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               m_state = e_line;
               *VALUE = new Falcon::Pseudo( line(), retval );
               return NUMERIC;
            }
            else
               m_string.append( chr );
         break;

         case e_floatNumber_e:
            if ( (chr < '0' || chr > '9' ) && chr != '+' && chr != '-' )
            {
               m_in->unget( chr );
               m_compiler->raiseError( e_inv_num_format, m_line );

               m_state = e_line;
               *VALUE = new Falcon::Pseudo( line(), 0.0 );
               return NUMERIC;
            }

            m_state = e_floatNumber_e1;
            m_string.append( chr );
         break;

         case e_floatNumber_e1:
            if ( chr < '0' || chr > '9' )
            {
               // end
               m_in->unget( chr );
               numeric retval;
               if ( ! m_string.parseDouble( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               m_state = e_line;
               *VALUE = new Falcon::Pseudo( line(), retval );
               return NUMERIC;
            }

            m_string.append( chr );
         break;

         case e_zeroNumber:
            if( chr == 'x' || chr == 'X' )
               m_state = e_hexNumber;
            else if ( chr >= '0' && chr <= '7' )
            {
               m_string.size( 0 );
               m_string.append( chr );
               m_state = e_octNumber;
            }
            else if ( chr == '.' ) {
               m_string = "0.";
               m_state = e_floatNumber;
            }
            else if ( isTokenLimit( chr ) )
            {
               m_state = e_line;
               *VALUE = new Falcon::Pseudo( line(), (int64) 0 );
               if( ! isWhiteSpace( chr ) )
                  m_in->unget( chr );
               return INTEGER;
            }
            else {
               m_compiler->raiseError( e_inv_num_format, m_line );
               m_state = e_line;
            }
         break;

         case e_octNumber:
            if ( chr < '0' || chr > '7' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! m_string.parseOctal( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               *VALUE = new Pseudo( line(), Pseudo::imm_int, retval );
               m_state = e_line;
               return INTEGER;
            }
            else
               m_string.append( chr );
         break;

         case e_hexNumber:
            if (  (chr >= '0' && chr <= '9') ||
                  (chr >= 'a' && chr <= 'f') ||
                  (chr >= 'A' && chr <= 'F')
               )
            {
                  m_string.append( chr );
            }
            else
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! m_string.parseHex( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               *VALUE = new Falcon::Pseudo( line(), Pseudo::imm_int, retval );
               m_state = e_line;
               return INTEGER;
            }
         break;

         case e_string:
            // an escape ?
            if ( chr == '\\' )
            {
               uint32 nextChar;
               m_in->readAhead( nextChar );
               switch ( nextChar )
               {
                  case '\\': nextChar = '\\'; break;

						case '\r':
							m_in->discardReadAhead( 1 );
							m_in->readAhead( nextChar );
							if( nextChar == '\n')
							{
								nextChar = ' ';
								m_line++;
								m_character = 0;
								m_bDirective = false;
							}
						break;

                  case '\n':
                     m_compiler->raiseError( e_eol_string );
                     nextChar = ' ';
                     m_line++;
                     m_character = 0;
                     m_bDirective = false;
                  break;

                  case '"': nextChar = '"'; break;
                  case 'n': nextChar = '\n'; break;
                  case 'b': nextChar = '\b'; break;
                  case 't': nextChar = '\t'; break;
                  case 'r': nextChar = '\r'; break;
                  case 's': nextChar = 1; break;

                  case 'x': case 'X':
                     nextChar = 1;
                     tempString.size(0);
                     m_state = e_stringHex;
                  break;

                  case '0':
                     nextChar = 1;
                     tempString.size(0);
                     m_state = e_stringOctal;
                  break;

                  default:
                     m_compiler->raiseError( e_inv_esc_sequence, m_line );
                     nextChar = 0;
               }
               if ( nextChar != 0 ) {
                  m_in->discardReadAhead( 1 );
                  if ( nextChar != 1 )
                     m_string.append( nextChar );
               }
            }
            else if ( chr == '\n' )
            {
               m_compiler->raiseError( e_eol_string );
               m_line++;
               m_character = 0;
               m_state = e_line;
               m_bDirective= false;
            }
            else if ( chr == '"' )
            {
               // closed string
               Falcon::String *str = m_module->addString( m_string );
               *VALUE = new Falcon::Pseudo( m_line, Falcon::Pseudo::imm_string, str );
               m_state = e_line;
               return STRING;
            }
            else
               m_string.append( chr );
         break;

         case e_stringHex:
            if (  (chr >= '0' && chr <= '9') ||
                     (chr >= 'a' && chr <= 'f') ||
                     (chr >= 'A' && chr <= 'F')
               )
            {
                  tempString.append( chr );
            }
            else
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! tempString.parseHex( retval ) || retval > 0xFFFFFFFF )
                  m_compiler->raiseError( e_inv_esc_sequence, m_line );
               m_string.append( (uint32) retval );
               m_state = e_string;
            }
         break;

         case e_stringOctal:
            if (  (chr >= '0' && chr <= '7') )
            {
                  tempString.append( chr );
            }
            else
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! tempString.parseOctal( retval ) || retval > 0xFFFFFFFF )
                  m_compiler->raiseError( e_inv_esc_sequence, m_line );
               m_string.append( (uint32) retval );
               m_state = e_string;
            }
         break;

         case e_directive:
         {
            if ( isTokenLimit( chr ) )
            {
               m_in->unget( chr );
               int result = checkDirectives();

               if ( result != 0 ){
                  m_state = e_line;
                  return result;
               }
            }
            else
               m_string.append( chr );
         }
         break;

         case e_word:
            if( isTokenLimit( chr )  )
            {
               // we are anyhow returning to line.
               m_in->unget( chr );
               m_state = e_line;

               // is this a label?
               if ( chr == ':' )
               {
                  *VALUE = new Falcon::Pseudo( m_line, m_compiler->addLabel( m_string ) );
                  return LABEL;
               }

               if ( m_bDirective )
               {
                  int token = checkPostDirectiveTokens();
                  if ( token != 0 ) {
                     return token;
                  }
               }
               else
               {
                  int token = checkTokens();
                  if ( token != 0 ) {
                     return token;
                  }
               }


               // is this a symbol?
               if ( m_string.getCharAt( 0 ) == '$' )
               {
                  Falcon::Symbol *sym = 0;
                  if ( m_string.length() > 1 )
                  {
                     String symname( m_string, 1, m_string.length() );
                     sym = m_compiler->findSymbol( symname );
                  }

                  if( sym == 0 ) {
                     m_compiler->raiseError( e_undef_sym, m_string );
                     *VALUE = regA_Inst();
                     return REG_A;
                  }

                  // return also if zero, as no one is gonna read it after an error.
                  *VALUE = new Falcon::Pseudo( m_line, Falcon::Pseudo::tsymbol, sym );
                  return SYMBOL;
               }

               // Is this a late binding?
               if ( m_string.getCharAt( 0 ) == '&' )
               {
                  if ( m_string.length() > 1 )
                  {
                     String symname( m_string, 1, m_string.length() );

                     // return also if zero, as no one is gonna read it after an error.
                     *VALUE = new Falcon::Pseudo( m_line,
                        Falcon::Pseudo::tlbind,
                        (Falcon::String *) m_compiler->addString(symname), false );
                     return SYMBOL;
                  }
                  else {
                     *VALUE = regA_Inst();
                     return REG_A;
                  }
               }

               // else, it's a name
               if ( m_bDirective )
               {
                  *VALUE = new Falcon::Pseudo( m_line, Falcon::Pseudo::imm_string, (String *) m_compiler->addString( m_string ) );
                  return NAME;
               }

               *VALUE = new Falcon::Pseudo( m_line, m_compiler->addLabel( m_string ) );
               return NAME;
            }
            else
               m_string.append( chr );

         break;
      }
   }

   return 0;
}

int AsmLexer::state_line( uint32 chr )
{
   if ( chr == '\n' )
   {
      m_line ++;
      m_character = 0;
      m_bDirective = false;
      return EOL;
   }
   else if ( chr == '.' && m_character == 1 )
   {
      m_state = e_directive;
      return 0;
   }
   else if ( chr < 0x20 )
   {
      // only invalid characters are in this range.
      String value;
      value.writeNumberHex( chr, true );
      m_compiler->raiseError( e_charRange, value, m_line );
      m_state = e_comment; // ignore the rest of the line
   }
   else if ( chr == ':' )
   {
      return COLON;
   }
   else if ( chr == '#' )
   {
      return e_stringID;
   }
   else if ( chr == '"' )
   {
      // we'll begin to read a string.
      m_state = e_string;
   }
   else if ( chr == '0' )
   {
      // we'll begin to read a 0 based number.
      m_state = e_zeroNumber;
   }
   else if ( chr >= '1' && chr <= '9' )
   {
      // reading a std number
      m_string.append( chr );
      m_state = e_intNumber;
   }
   else if ( chr == '-' )
   {
      // this may be a number;
      m_string.append( chr );
      uint32 nextChr;
      m_in->readAhead( nextChr );
      if( nextChr >= '0' && nextChr <= '9' )
      {
         m_in->discardReadAhead( 1 );
         m_string.append( nextChr );
         m_state = e_intNumber;
      }
      else if ( nextChr != '-' && nextChr != '=' ) {
         m_compiler->raiseError( e_inv_num_format );
      }
   }
   else if ( chr == ';' )
   {
      m_state = e_comment;
      return 0;
   }
   else if ( chr == ',' )
   {
      return COMMA;
   }
   else {
      // enter in word state
      m_state = e_word;
      m_string.append( chr );
   }

   return 0;
}


int AsmLexer::checkTokens()
{
   switch( m_string.length() )
   {
      case 1:
         if ( m_string.compareIgnoreCase( "A" ) == 0 )
         {
            *VALUE = regA_Inst();
            return REG_A;
         }
         if ( m_string.compareIgnoreCase( "B" ) == 0 )
         {
            *VALUE = regB_Inst();
            return REG_B;
         }
         if ( m_string.compareIgnoreCase( "T" ) == 0 )
         {
            *VALUE = true_Inst();
            return TRUE_TOKEN;
         }
         if ( m_string.compareIgnoreCase( "F" ) == 0 )
         {
            *VALUE = false_Inst();
            return FALSE_TOKEN;
         }
      break;

      case 2:
         if ( m_string.compareIgnoreCase( "LD" ) == 0 )
            return I_LD;
         else if ( m_string.compareIgnoreCase( "EQ" ) == 0 )
            return I_EQ;
         else if ( m_string.compareIgnoreCase( "GT" ) == 0 )
            return I_GT;
         else if ( m_string.compareIgnoreCase( "GE" ) == 0 )
            return I_GE;
         else if ( m_string.compareIgnoreCase( "LT" ) == 0 )
            return I_LT;
         else if ( m_string.compareIgnoreCase( "LE" ) == 0 )
            return I_LE;
         else if ( m_string.compareIgnoreCase( "OR" ) == 0 )
            return I_OR;
         else if ( m_string.compareIgnoreCase( "IN" ) == 0 )
            return I_IN;
         else if ( m_string.compareIgnoreCase( "S1" ) == 0 )
         {
            *VALUE = regS1_Inst();
            return REG_S1;
         }
         else if ( m_string.compareIgnoreCase( "S2" ) == 0 )
         {
            *VALUE = regS2_Inst();
            return REG_S2;
         }
         else if ( m_string.compareIgnoreCase( "L1" ) == 0 )
         {
            *VALUE = regL1_Inst();
            return REG_L1;
         }
         else if ( m_string.compareIgnoreCase( "L2" ) == 0 )
         {
            *VALUE = regL2_Inst();
            return REG_L2;
         }
      break;

      case 3:
         if ( m_string.compareIgnoreCase( "NIL" ) == 0 )
         {
            *VALUE = nil_Inst();
            return NIL;
         }
         else if ( m_string.compareIgnoreCase( "ADD" ) == 0 )
             return I_ADD;
         else if ( m_string.compareIgnoreCase( "AND" ) == 0 )
             return I_AND;
         else if ( m_string.compareIgnoreCase( "SUB" ) == 0 )
             return I_SUB;
         else if ( m_string.compareIgnoreCase( "MUL" ) == 0 )
             return I_MUL;
         else if ( m_string.compareIgnoreCase( "DIV" ) == 0 )
             return I_DIV;
         else if ( m_string.compareIgnoreCase( "POW" ) == 0 )
             return I_POW;
         else if ( m_string.compareIgnoreCase( "INC" ) == 0 )
             return I_INC;
         else if ( m_string.compareIgnoreCase( "DEC" ) == 0 )
             return I_DEC;
         else if ( m_string.compareIgnoreCase( "NEG" ) == 0 )
             return I_NEG;
         else if ( m_string.compareIgnoreCase( "NOT" ) == 0 )
             return I_NOT;
         else if ( m_string.compareIgnoreCase( "RET" ) == 0 )
             return I_RET;
         else if ( m_string.compareIgnoreCase( "POP" ) == 0 )
             return I_POP;
         else if ( m_string.compareIgnoreCase( "LDP" ) == 0 )
             return I_LDP;
         else if ( m_string.compareIgnoreCase( "STP" ) == 0 )
             return I_STP;
         else if ( m_string.compareIgnoreCase( "LDV" ) == 0 )
             return I_LDV;
         else if ( m_string.compareIgnoreCase( "JMP" ) == 0 )
             return I_JMP;
         else if ( m_string.compareIgnoreCase( "IFT" ) == 0 )
             return I_IFT;
         else if ( m_string.compareIgnoreCase( "IFF" ) == 0 )
             return I_IFF;
         else if ( m_string.compareIgnoreCase( "NEQ" ) == 0 )
             return I_NEQ;
         else if ( m_string.compareIgnoreCase( "TRY" ) == 0 )
             return I_TRY;
         else if ( m_string.compareIgnoreCase( "RIS" ) == 0 )
             return I_RIS;
         else if ( m_string.compareIgnoreCase( "BOR" ) == 0 )
             return I_BOR;
         else if ( m_string.compareIgnoreCase( "ORS" ) == 0 )
             return I_ORS;
         else if ( m_string.compareIgnoreCase( "HAS" ) == 0 )
             return I_HAS;
         else if ( m_string.compareIgnoreCase( "END" ) == 0 )
             return I_END;
         else if ( m_string.compareIgnoreCase( "SHL" ) == 0 )
             return I_SHL;
         else if ( m_string.compareIgnoreCase( "SHR" ) == 0 )
             return I_SHR;
         else if ( m_string.compareIgnoreCase( "LSB" ) == 0 )
             return I_LSB;
         else if ( m_string.compareIgnoreCase( "STV" ) == 0 )
             return I_STV;
         else if ( m_string.compareIgnoreCase( "WRT" ) == 0 )
             return I_WRT;
         else if ( m_string.compareIgnoreCase( "MOD" ) == 0 )
             return I_MOD;
         else if ( m_string.compareIgnoreCase( "STO" ) == 0 )
             return I_STO;
      break;

     case 4:
         if ( m_string.compareIgnoreCase( "LNIL" ) == 0 )
             return I_LNIL;
         if ( m_string.compareIgnoreCase( "ADDS" ) == 0 )
             return I_ADDS;
         if ( m_string.compareIgnoreCase( "SUBS" ) == 0 )
             return I_SUBS;
         if ( m_string.compareIgnoreCase( "MULS" ) == 0 )
             return I_MULS;
         if ( m_string.compareIgnoreCase( "DIVS" ) == 0 )
             return I_DIVS;
         if ( m_string.compareIgnoreCase( "POWS" ) == 0 )
             return I_POWS;
         if ( m_string.compareIgnoreCase( "RETV" ) == 0 )
             return I_RETV;
         if ( m_string.compareIgnoreCase( "RETA" ) == 0 )
             return I_RETA;
         if ( m_string.compareIgnoreCase( "FORK" ) == 0 )
             return I_FORK;
         if ( m_string.compareIgnoreCase( "PUSH" ) == 0 )
             return I_PUSH;
         if ( m_string.compareIgnoreCase( "PSHN" ) == 0 )
             return I_PSHN;
         if ( m_string.compareIgnoreCase( "XPOP" ) == 0 )
             return I_XPOP;
         if ( m_string.compareIgnoreCase( "LDVT" ) == 0 )
             return I_LDVT;
         if ( m_string.compareIgnoreCase( "STVR" ) == 0 )
             return I_STVR;
         if ( m_string.compareIgnoreCase( "STVS" ) == 0 )
             return I_STVS;
         if ( m_string.compareIgnoreCase( "LDPT" ) == 0 )
             return I_LDPT;
         if ( m_string.compareIgnoreCase( "STPR" ) == 0 )
             return I_STPR;
         if ( m_string.compareIgnoreCase( "STPS" ) == 0 )
             return I_STPS;
         if ( m_string.compareIgnoreCase( "TRAV" ) == 0 )
             return I_TRAV;
         if ( m_string.compareIgnoreCase( "INCP" ) == 0 )
             return I_INCP;
         if ( m_string.compareIgnoreCase( "DECP" ) == 0 )
             return I_DECP;
         if ( m_string.compareIgnoreCase( "TRAN" ) == 0 )
             return I_TRAN;
         if ( m_string.compareIgnoreCase( "TRAL" ) == 0 )
             return I_TRAL;
         if ( m_string.compareIgnoreCase( "IPOP" ) == 0 )
             return I_IPOP;
         if ( m_string.compareIgnoreCase( "GENA" ) == 0 )
             return I_GENA;
         if ( m_string.compareIgnoreCase( "GEND" ) == 0 )
             return I_GEND;
         if ( m_string.compareIgnoreCase( "GENR" ) == 0 )
             return I_GENR;
         if ( m_string.compareIgnoreCase( "GEOR" ) == 0 )
             return I_GEOR;
         if ( m_string.compareIgnoreCase( "BOOL" ) == 0 )
             return I_BOOL;
         if ( m_string.compareIgnoreCase( "UNPK" ) == 0 )
             return I_UNPK;
         if ( m_string.compareIgnoreCase( "UNPS" ) == 0 )
             return I_UNPS;
         if ( m_string.compareIgnoreCase( "PSHR" ) == 0 )
             return I_PSHR;
         if ( m_string.compareIgnoreCase( "SWCH" ) == 0 )
             return I_SWCH;
         if ( m_string.compareIgnoreCase( "SELE" ) == 0 )
             return I_SELE;
         if ( m_string.compareIgnoreCase( "PTRY" ) == 0 )
             return I_PTRY;
         if ( m_string.compareIgnoreCase( "JTRY" ) == 0 )
             return I_JTRY;
         if ( m_string.compareIgnoreCase( "INST" ) == 0 )
             return I_INST;
         if ( m_string.compareIgnoreCase( "LDRF" ) == 0 )
             return I_LDRF;
         if ( m_string.compareIgnoreCase( "ONCE" ) == 0 )
             return I_ONCE;
         if ( m_string.compareIgnoreCase( "BAND" ) == 0 )
             return I_BAND;
         if ( m_string.compareIgnoreCase( "BXOR" ) == 0 )
             return I_BXOR;
         if ( m_string.compareIgnoreCase( "BNOT" ) == 0 )
             return I_BNOT;
         if ( m_string.compareIgnoreCase( "MODS" ) == 0 )
             return I_MODS;
         if ( m_string.compareIgnoreCase( "ANDS" ) == 0 )
             return I_ANDS;
         if ( m_string.compareIgnoreCase( "XORS" ) == 0 )
             return I_XORS;
         if ( m_string.compareIgnoreCase( "NOTS" ) == 0 )
             return I_NOTS;
         if ( m_string.compareIgnoreCase( "HASN" ) == 0 )
             return I_HASN;
         if ( m_string.compareIgnoreCase( "GIVE" ) == 0 )
             return I_GIVE;
         if ( m_string.compareIgnoreCase( "GIVN" ) == 0 )
             return I_GIVN;
         if ( m_string.compareIgnoreCase( "NOIN" ) == 0 )
             return I_NOIN;
         if ( m_string.compareIgnoreCase( "PROV" ) == 0 )
             return I_PROV;
         if ( m_string.compareIgnoreCase( "PEEK" ) == 0 )
             return I_PEEK;
         if ( m_string.compareIgnoreCase( "PSIN" ) == 0 )
             return I_PSIN;
         if ( m_string.compareIgnoreCase( "PASS" ) == 0 )
             return I_PASS;
         if ( m_string.compareIgnoreCase( "SHLS" ) == 0 )
             return I_SHLS;
         if ( m_string.compareIgnoreCase( "SHRS" ) == 0 )
             return I_SHRS;
         if ( m_string.compareIgnoreCase( "LDVR" ) == 0 )
             return I_LDVR;
         if ( m_string.compareIgnoreCase( "LDPR" ) == 0 )
             return I_LDPR;
         if ( m_string.compareIgnoreCase( "CALL" ) == 0 )
             return I_CALL;
         if ( m_string.compareIgnoreCase( "INDI" ) == 0 )
             return I_INDI;
         if ( m_string.compareIgnoreCase( "STEX" ) == 0 )
             return I_STEX;
         if ( m_string.compareIgnoreCase( "TRAC" ) == 0 )
             return I_TRAC;
         if ( m_string.compareIgnoreCase( "FORB" ) == 0 )
             return I_FORB;
      break;

   }

   return 0;
}


int AsmLexer::checkDirectives()
{

   m_bDirective = true;
   switch( m_string.length() )
   {
      case 3:
         if ( m_string == "var" )
            return DVAR;
         if ( m_string == "has" )
            return DHAS;
      break;

      case 4:
         if ( m_string == "load" )
            return DLOAD;
         if ( m_string == "line" )
            return DLINE;
         if ( m_string == "func" )
            return DFUNC;
         if ( m_string == "prop" )
            return DPROP;
         if ( m_string == "ctor" )
            return DCTOR;
         if ( m_string == "from" )
            return DFROM;
         if ( m_string == "case" )
         {
            // cases behave as instructions
            m_bDirective = false;
            return DCASE;
         }
      break;

      case 5:
         if ( m_string == "const" )
            return DCONST;
         if ( m_string == "local" )
            return DLOCAL;
         if ( m_string == "param" )
            return DPARAM;
         if ( m_string == "class" )
            return DCLASS;
         if ( m_string == "entry" )
            return DENTRY;
         if ( m_string == "hasnt" )
            return DHASNT;
      break;

      case 6:
         if ( m_string == "global" )
            return DGLOBAL;
         if ( m_string == "import" )
            return DIMPORT;
         if ( m_string == "attrib" )
            return DATTRIB;
         if ( m_string == "method" )
            return DMETHOD;
         if ( m_string == "extern" )
            return DEXTERN;
         if ( m_string == "module" )
            return DMODULE;
         if ( m_string == "switch" )
         {
            // switch behaves as an instruction
            m_bDirective = false;
            return DSWITCH;
         }
         if ( m_string == "select" )
         {
            // switch behaves as an instruction
            m_bDirective = false;
            return DSELECT;
         }
         if ( m_string == "string" )
            return DSTRING;
      break;

      case 7:
         if ( m_string == "funcdef" )
            return DFUNCDEF;
         if ( m_string == "propref" )
            return DPROPREF;
         if ( m_string == "endfunc" )
            return DENDFUNC;
         if ( m_string == "inherit" )
            return DINHERIT;
         if ( m_string == "istring" )
            return DISTRING;
      break;

      case 8:
         if ( m_string == "classdef" )
            return DCLASSDEF;
         if ( m_string == "endclass" )
            return DENDCLASS;
         if ( m_string == "instance" )
            return DINSTANCE;
      break;

      case 9:
         if ( m_string == "endswitch" )
            return DENDSWITCH;
      break;

   }

   m_compiler->raiseError( e_inv_direct, m_string );
   m_state = e_comment;
   m_bDirective = false;
   return 0;
}

int AsmLexer::checkPostDirectiveTokens()
{
   if ( m_string.compareIgnoreCase( "EXPORT" ) == 0 )
      return EXPORT;
   if ( m_string.compare( "NIL" ) == 0 )
   {
      *VALUE = nil_Inst();
      return NIL;
   }

   return 0;
}

}


/* end of fasm_lexer.cpp */
