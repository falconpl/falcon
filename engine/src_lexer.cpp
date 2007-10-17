/*
   FALCON - The Falcon Programming Language
   FILE: src_lexer.cpp
   $Id: src_lexer.cpp,v 1.21 2007/08/19 07:25:53 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 26 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Short description
*/
#include <stdio.h>

#include <falcon/error.h>
#include <falcon/setup.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/compiler.h>
#include <falcon/transcoding.h>

#include <falcon/stdstreams.h>

// Token internally used
#define ITOK_SEMICOMMA  0x7FFFFF01

#define VALUE  ((YYSTYPE*)value())


// Get the bison generated token defs
#include "src_parser.h"

namespace Falcon {

SrcLexer::SrcLexer( Compiler *comp, Stream *in ):
   m_prevStat(0),
   m_line( 1 ),
   m_previousLine( 1 ),
   m_value(0),
   m_in( in ),
   m_compiler( comp ),
   m_contexts(0),
   m_squareContexts(0),
   m_firstEq( true ),
   m_character( 0 ),
   m_state( e_line ),
   m_done( false ),
   m_firstSym( true ),
   m_addEol( false ),
   m_lineFilled( false ),
   m_mode( t_mNormal )
{}

/*
int SrcLexer::lex()
{
   char buf[128];
   int ret = lex_();
   m_string.toCString( buf, 128 );
   printf( "Returning token %d (%s)\n", ret, buf );
   return ret;
}
*/

int SrcLexer::lex()
{
   switch( m_mode )
   {
      case t_mNormal: return lex_normal();
      case t_mOutscape: return lex_outscape();
      case t_mEval: return lex_eval();
   }

   return 0;
}


int SrcLexer::lex_outscape()
{
   enum {
      e_normal,
      e_esc1,
      e_esc2,
      e_escF,
      e_escA,
      e_escL,
   }
   state;

   // clear string
   m_string.size( 0 );

   // in outscape mode, everything up to an escape (or at EOF) is a "fast print" statement.
   uint32 chr;
   state = e_normal;
   m_state = e_line;

   while( m_mode == t_mOutscape && m_in->get( chr ) )
   {
      if ( chr == '\n' )
      {
         m_previousLine = m_line;
         m_line++;
         m_character = 0;
      }
      else
      {
         m_character++;
      }

      switch( state )
      {
         case e_normal:
            if ( chr == '<' )
               state = e_esc1;
            else
               m_string.append( chr );
         break;

         case e_esc1:
            if ( chr == '?' )
               state = e_esc2;
            else {
               m_string.append( '<' );
               m_string.append( chr );
               state  = e_normal;
            }
         break;

         case e_esc2:
            if ( chr == '=' )
            {
               // we enter now in the eval mode
               m_mode = t_mEval;
               // and break from the loop so to return the string to print.
               break;
            }
            else if ( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
            {
               // we enter now the normal mode; we start to consider a standard program.
               m_mode = t_mNormal;
               // and break from the loop so to return the string to print.
               break;
            }
            else if ( chr == 'f' )
            {
               state = e_escF;
            }
            else
            {
               state = e_normal;
               m_string.append( '<' );
               m_string.append( '?' );
               m_string.append( chr );
            }
         break;

         case e_escF:
            if ( chr == 'a' )
            {
               state = e_escA;
            }
            else {
               state = e_normal;
               m_string.append( '<' );
               m_string.append( '?' );
               m_string.append( 'f' );
               m_string.append( chr );
            }
         break;

         case e_escA:
            if ( chr == 'l' )
            {
              state = e_escL;
            }
            else {
               state = e_normal;
               m_string.append( '<' );
               m_string.append( '?' );
               m_string.append( 'f' );
               m_string.append( 'a' );
               m_string.append( chr );
            }
         break;

         case e_escL:
            if ( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
            {
               // we enter now the normal mode; we start to consider a standard program.
               m_mode = t_mNormal;
               // and break from the loop so to return the string to print.
               break;
            }
            else {
               state = e_normal;
               m_string.append( '<' );
               m_string.append( '?' );
               m_string.append( 'f' );
               m_string.append( 'a' );
               m_string.append( 'l' );
               m_string.append( chr );
            }
         break;
      }
   }

   if( m_string.size() != 0 )
   {
      VALUE->stringp = m_compiler->addString( m_string );
      m_string.size( 0 );
      return OUTER_STRING;
   }

   // else proceed with normal evaluation as we can't return nothing
   if( m_in->good() )
   {
      switch( m_mode )
      {
         case t_mNormal: return lex_normal();
         case t_mEval: return lex_eval();
      }
   }

   return 0;
}


int SrcLexer::lex_eval()
{
   // prepare for a normal scan
   m_mode = t_mNormal;

   // returns an LT, which will be interpreted as a fast print
   return LT;
}


int SrcLexer::lex_normal()
{
   // generate an extra eol?
   if ( m_addEol )
   {
      m_firstEq = true;
      m_addEol = false;
      m_lineFilled = false;
      return EOL;
   }

   if ( m_done )
      return 0;

   // check for shell directive
   if ( m_line == 1 && m_character == 0 )
   {
      uint32 ch1, ch2;
      if( m_in->get( ch1 ) )
      {
         if( m_in->get( ch2 ) )
         {
            if ( ch1 == '#' && ch2 == '!' )
            {
               while( ch1 != '\n' )
                  if( ! m_in->get( ch1 ) )
                     return 0;
               m_line++;
            }
            else {
               m_in->unget( ch2 );
               m_in->unget( ch1 );
            }
         }
         else
            m_in->unget( ch1 );
      }
      else
         return 0;
   }

   // reset previous token
   m_string.size(0);
   m_string.manipulator( &csh::handler_buffer );

   String tempString;
   uint32 chr;

   bool next_loop = true;

   while( next_loop )
   {
      next_loop = m_in->get( chr );

      if ( ! next_loop )
      {
         // fake an empty terminator at the end of input.
         chr = '\n';
         m_done = true;
         m_addEol = true;
      }

      m_character++;

      switch ( m_state )
      {
         case e_line:
            // in none status, we have to discard blanks and even ignore '\n';
            // we enter in line or string status depending on what we find.
            if( ! isWhiteSpace( chr ) )
            {
               m_previousLine = m_line;
               // whitespaces and '\n' can't follow a valid symbol,
               // as since they are token limiters, and they would be read
               // ahead after token begin, valid symbols and token has already
               // been returned.

               int token = state_line( chr );
               if ( token != 0 ) {
                  // we have a token in this line
                  //m_lineFilled = true;
                  return token;
               }
            }
         break;

         case e_symbol:
            m_lineFilled = true;
            if ( isTokenLimit( chr ) )
            {
               // end of symbol

               // push this chr back; we want to read it again in line state
               if( ! isWhiteSpace( chr ) ) // save a loop
                  m_in->unget( chr );

               m_state = e_line;

               // it may be a named token
               int token = checkLimitedTokens();

               // we have a first symbol, that is, can't be a directive anymore
               m_firstSym = false;

               if ( token != 0 )
               {
                  return token;
               }

               // see if we have named constants
               const Value *cval = m_compiler->getConstant( m_string );

               if ( cval != 0 ) {
                  switch( cval->type() ) {
                     case Falcon::Value::t_nil: return NIL;
                     case Falcon::Value::t_imm_integer: VALUE->integer = cval->asInteger(); return INTNUM;
                     case Falcon::Value::t_imm_num: VALUE->numeric = cval->asNumeric(); return DBLNUM;
                     case Falcon::Value::t_imm_string: VALUE->stringp = cval->asString(); return STRING;
                     default: return NIL;  // signal error?
                  }
               }

               // we have a symbol
               VALUE->stringp = m_compiler->addString( m_string );
               return SYMBOL;
            }
            else
               m_string.append( chr );
         break;

         case e_operator:
         {
            // have we a token?
            int token = checkUnlimitedTokens( chr );
            if ( token > 0 ) {
               // great, we have a token
               m_lineFilled = true;
               m_state = e_line;
               
               // a bit of galore: discard extra "\n" or "\r\n" in case of outer escape
               if ( m_mode == t_mOutscape )
               {
                  if ( chr != '\n' ) {
                     
                     // ok, not a newline; but is it an INET newline?
                     if ( chr == '\r') 
                     {
                        uint32 ch1 = 0;
                        m_in->get( ch1 );
                        if ( ch1 != '\n' )
                        {
                           m_in->unget( ch1 );
                           m_in->unget( chr );
                        }
                        // else silently discard
                     }
                     else 
                        m_in->unget( chr );
                  }
                  
               }
               else
                  m_in->unget( chr );
               
               return token;
            }
            else if ( token < 0 || m_string.length() == 3 ) {
               // We have aknowledged this can't be a valid token.
               m_in->unget( chr );
               m_state = e_line;
               m_compiler->raiseError( e_inv_token, m_string, m_line );
               // do not return.
            }

            // if we have been switched to another state, we have aknowledged something
            if ( m_state != e_operator ) {
               m_in->unget( chr );
               m_string.size( 0 );
            }
            else
               m_string.append( chr );
         }
         break;

         case e_eolCommentString:
         case e_eolComment:
            if ( chr == '\n' )
            {
               m_previousLine = m_line;
               m_line ++;
               m_character = 0;
               m_state = (m_state == e_eolCommentString) ? e_postString: e_line;

               // a real EOL has been provided here.
               if ( m_state == e_line )
               {
                  m_firstSym = true;
                  m_firstEq = true;
                  if ( m_lineFilled )
                  {
                     m_lineFilled = false;
                     return EOL;
                  }
               }
            }
         break;

         case e_blockCommentString:
         case e_blockComment:
            if ( chr == '\n' )
            { // previous line stays the same
               m_line ++;
               m_character = 0;
            }
            else if ( chr == '*' )
            {
               uint32 nextChr;
               m_in->readAhead( nextChr );
               if ( nextChr == '/' )
               {
                  m_in->discardReadAhead( nextChr );
                  m_state = m_state == e_blockCommentString ? e_postString : e_line;
               }
            }
         break;

         case e_intNumber:
            m_lineFilled = true;
            if ( chr == '.' )
            {
               m_state = e_floatNumber;
            }
            else if ( chr == 'e' )
            {
               m_state = e_floatNumber_e;
            }
            else if ( chr < '0' || chr > '9' )
            {
               // end
               m_in->unget( chr );
               int64 retval;
               if ( ! m_string.parseInt( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->integer = retval;
               m_state = e_line;
               return INTNUM;
            }

            m_string.append( chr );
         break;

         case e_floatNumber:
            m_lineFilled = true;
            if ( chr == 'e' )
            {
               m_state = e_floatNumber_e;
               m_string.append( chr );
            }
            else if ( (chr < '0' || chr > '9')  && chr != '.'  )
            {
               // end
               m_in->unget( chr );
               numeric retval;
               if ( ! m_string.parseDouble( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->numeric = retval;
               m_state = e_line;
               return DBLNUM;
            }
            else
               m_string.append( chr );
         break;

         case e_floatNumber_e:
            m_lineFilled = true;
            if ( (chr < '0' || chr > '9' ) && chr != '+' && chr != '-' )
            {
               m_in->unget( chr );
               m_compiler->raiseError( e_inv_num_format, m_line );

               m_state = e_line;
               VALUE->numeric = 0.0;
               return DBLNUM;
            }

            m_state = e_floatNumber_e1;
            m_string.append( chr );
         break;

         case e_floatNumber_e1:
            m_lineFilled = true;
            if ( chr < '0' || chr > '9' )
            {
               // end
               m_in->unget( chr );
               numeric retval;
               if ( ! m_string.parseDouble( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               m_state = e_line;
               VALUE->numeric = retval;
               return DBLNUM;
            }

            m_string.append( chr );
         break;

         case e_zeroNumber:
            m_lineFilled = true;
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
               VALUE->integer = 0;
               if( ! isWhiteSpace( chr ) )
                  m_in->unget( chr );
               return INTNUM;
            }
            else {
               m_compiler->raiseError( e_inv_num_format, m_line );
               m_state = e_line;
            }
         break;

         case e_octNumber:
            m_lineFilled = true;
            if ( chr < '0' || chr > '7' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! m_string.parseOctal( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->integer = retval;
               m_state = e_line;
               return INTNUM;
            }
            else
               m_string.append( chr );
         break;

         case e_hexNumber:
            m_lineFilled = true;
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

               VALUE->integer = retval;
               m_state = e_line;
               return INTNUM;
            }
         break;

         case e_litString:
            m_lineFilled = true;
            // an escape ?
            if ( chr == '\\' )
            {
               uint32 nextChar;
               m_in->readAhead( nextChar );
               if( nextChar == '\'' )
               {

                  m_in->discardReadAhead( 1 );
                  m_string.append( '\'' );
               }
               else {
                  m_string.append( '\\' );
               }
            }
            else if ( chr == '\n' )
            {
               m_previousLine = m_line;
               m_line++;
               m_lineFilled = false;
               m_character = 0;
               m_compiler->raiseError( e_nl_in_lit, m_previousLine );
               m_state = e_line;
            }
            else if ( chr == '\'' )
            {
               m_state = e_line;
               VALUE->stringp = m_compiler->addString( m_string );
               return STRING;
            }
            else
               m_string.append( chr );
         break;

         case e_string:
            m_lineFilled = true;
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
                        m_previousLine = m_line;
                        m_line++;
                        m_character = 0;
                        m_state = e_stringRunning;
                     }
                  break;

                  case '\n':
                     nextChar = ' ';
                     m_previousLine = m_line;
                     m_line++;
                     m_character = 0;
                     m_state = e_stringRunning;
                  break;

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

                  // when none of the above, just keep nextchar as is.
               }
               if ( nextChar != 0 ) {
                  m_in->discardReadAhead( 1 );
                  if ( nextChar != 1 )
                     m_string.append( nextChar );
               }
            }
            else if ( chr == '\n' )
            {
               m_string.append( '\n' );
               m_previousLine = m_line;
               m_line++;
               m_character = 0;
            }
            else if ( chr == m_chrEndString )
            {
               // closed string
               m_state = e_postString;
            }
            else
               m_string.append( chr );
         break;

         case e_stringRunning:
            if ( ! isWhiteSpace( chr ) )
            {
               if ( chr == '\n' )
               {
                  m_previousLine = m_line;
                  m_line++;
                  m_character = 0;
               }
               else {
                  m_state = e_string;
                  m_string.append( chr );
               }
            }
         break;

         case e_postString:
            if ( chr == '\n' ) {
               // don't change previous line
               m_line++;
               m_character = 0;
               m_addEol = true; // force to generate an EOL for the parser
               // end of input?
               if ( m_done ) {
                  VALUE->stringp = m_compiler->addString( m_string );
                  m_state = e_line;
                  return STRING;
               }
            }
            else if ( chr == '"' )
            {
               // a chained string.
               m_addEol = false;
               m_state = e_string;
               m_chrEndString = '"'; //... up to the matching "
            }
            else if ( chr == 0x201C )
            {
               // a chained string.
               m_addEol = false;
               m_state = e_string;
               m_chrEndString = 0x201D; //... up to the matching close quote
            }
            else if ( chr == 0x300C )
            {
               // a chained string.
               m_addEol = false;
               m_state = e_string;
               m_chrEndString = 0x300D; //... up to the matching close japanese quote
            }
            else if ( chr == '/' )
            {
               // may be a comment.
               uint32 chr1;
               if ( ! m_in->get( chr1 ) )
               {
                  // end of input.
                  m_done = true;
                  VALUE->stringp = m_compiler->addString( m_string );
                  return STRING;
               }

               if ( chr1 == '*' )
               {
                  m_state = e_blockCommentString;
               }
               else if ( chr1 == '/' ) {
                  VALUE->stringp = m_compiler->addString( m_string );
                  m_state = e_eolComment;
                  return STRING;
               }
               else {
                  // not a comment
                  m_in->unget( chr1 );
                  m_in->unget( chr );
                  VALUE->stringp = m_compiler->addString( m_string );
                  m_state = e_line;
                  return STRING;
               }
            }
            else if ( ! isWhiteSpace( chr ) )
            {

               m_in->unget( chr );
               VALUE->stringp = m_compiler->addString( m_string );
               m_state = e_line;
               return STRING;
            }


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
      }
   }

   return 0;
}

int SrcLexer::state_line( uint32 chr )
{
   if ( chr == '\n' )
   {
      m_previousLine = m_line;
      m_line ++;
      m_character = 0;

      // a real EOL has been provided here.
      m_firstEq = true;
      m_firstSym = true;
      if ( m_lineFilled )
      {
         m_lineFilled = false;
         return EOL;
      }
   }
   else if ( chr == '\\' )
   {
      // don't return at next eol:
      uint32 nextChr;
      m_in->readAhead( nextChr );
      if ( nextChr == '\n' )
      {
         m_previousLine = m_line;
         m_line ++;
         m_character = 0;
         m_in->discardReadAhead( 1 );
      }
      else if ( nextChr == '\r' )
      {
         // discard next char
         m_in->discardReadAhead( 1 );
         m_in->readAhead( nextChr );
         if ( nextChr == '\n' )
         {
            m_previousLine = m_line;
            m_line++;
            m_character = 0;
            m_in->discardReadAhead( 2 );
         }
      }

   }
   else if ( chr < 0x20 )
   {
      // only invalid characters are in this range.
      String value;
      value.writeNumberHex( chr, true );
      m_compiler->raiseError( e_charRange, value, m_line );
      m_state = e_eolComment; // ignore the rest of the line
   }
   else if ( chr == '"' )
   {
      // we'll begin to read a string.
      m_state = e_string;
      m_chrEndString = '"'; //... up to the matching "
   }
   else if ( chr == 0x201C )
   {
      // we'll begin to read a string.
      m_state = e_string;
      m_chrEndString = 0x201D; //... up to the matching close quote
   }
   else if ( chr == 0x300C )
   {
      // we'll begin to read a string.
      m_state = e_string;
      m_chrEndString = 0x300D; //... up to the matching close japanese quote
   }
   else if ( chr == '\'' )
   {
      // we'll begin to read a non escaped string
      m_state = e_litString;
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
   else
   {
      // store this character.
      m_string.append( chr );

      // if it's an operator character, enter in operator mode.
      if ( isTokenLimit( chr ) )
         m_state = e_operator;
      else
         m_state = e_symbol;
   }

   return 0;
}

int SrcLexer::checkUnlimitedTokens( uint32 nextChar )
{
   switch( m_string.length() )
   {
      case 1:
      {
         uint32 chr = m_string.getCharAt( 0 );
         if ( chr == ';'  )
         {
            m_firstEq = true;
            // but not first sym
            if ( m_lineFilled )
            {
               m_lineFilled = false;
               return EOL;
            }
         }
         else if ( chr == '+' && nextChar != '=' && nextChar != '+' )
            return PLUS;
         else if ( chr == '-' && nextChar != '=' && nextChar != '-' )
            return MINUS;
         else if ( chr == '*' && nextChar != '=' && nextChar != '*' )
            return STAR;
         else if ( chr == '/' && nextChar != '=' && nextChar != '/' && nextChar != '*' )
            return SLASH;
         else if ( chr == '%' && nextChar != '=' )
            return PERCENT;
         else if ( chr == '&' && nextChar != '=' )
            return AMPER;
         else if ( chr == '|' && nextChar != '=' )
            return VBAR;
         else if ( chr == '^' && nextChar != '=' )
            return CAP;
         else if ( chr == '!' && nextChar != '=' )
               return BANG;

         else if ( chr == '$' )
         {
            return DOLLAR;
         }
         else if ( chr == ':' )
         {
            // expressions after : are reassigned, but only out of contexts
            if( m_contexts == 0 && m_squareContexts == 0 )
               m_firstEq = true;
               // but they don't reset first sym
            return COLON;
         }
         else if( chr == ',' )
            return COMMA;
         else if( chr == ';' )
            return ITOK_SEMICOMMA;
         else if ( chr == '.' && nextChar != '=' )
            return DOT;
         else if ( chr == '?' )
         {
            if ( ! parsingFtd() || nextChar != '>' )
               return QUESTION;
         }
         else if ( chr == '>' && nextChar != '=' && nextChar != '>' )
            return GT;
         else if ( chr == '<' && nextChar != '=' && nextChar != '<' )
            return LT;
         else if ( chr == '=' && nextChar != '=' && nextChar != '>' )
         {
            if ( m_firstEq ) {
               m_firstEq = false;
               return OP_ASSIGN;
            }
            else
               return OP_EQ;
         }
         else if ( chr == '(' || chr == 0xff08 )
         {
            m_contexts++;
            return OPENPAR;
         }
         else if ( chr == ')' || chr == 0xff09 )
         {
            if ( m_contexts == 0 )
               m_compiler->raiseError( e_par_close_unbal, m_line );
            else
            {
               m_contexts--;
               return CLOSEPAR;
            }
         }
         else if ( chr == '[' )
         {
            m_squareContexts++;
            return OPENSQUARE;
         }
         else if ( chr == ']' )
         {
            if ( m_squareContexts == 0 )
            {
               m_compiler->raiseError( e_square_close_unbal, m_line );
            }
            else
            {
               m_squareContexts--;
               return CLOSESQUARE;
            }
         }
         else if ( chr == '@' )
         {
            return ATSIGN;
         }
         else if ( chr == '#' )
         {
            return DIESIS;
         }
      }
      break;

      case 2:
         if ( m_string == "=>" )
            return ARROW;
         else if ( m_string == "==" )
            return EEQ;
         else if ( m_string == "!=" )
            return NEQ;
         else if ( m_string == ">=" )
            return GE;
         else if ( m_string == "<=" )
            return LE;
         else if ( m_string == "+=" )
            return ASSIGN_ADD;
         else if ( m_string == "-=" )
            return ASSIGN_SUB;
         else if ( m_string == "*=" )
            return ASSIGN_MUL;
         else if ( m_string == "/=" )
            return ASSIGN_DIV;
         else if ( m_string == "%=" )
            return ASSIGN_MOD;
         else if ( m_string == "&=" )
            return ASSIGN_BAND;
         else if ( m_string == "|=" )
            return ASSIGN_BOR;
         else if ( m_string == "^=" )
            return ASSIGN_BXOR;
         else if ( m_string == ">>" && nextChar != '=' )
            return SHR;
         else if ( m_string == "<<" && nextChar != '=' )
            return SHL;
         else if ( m_string == "++" )
            return INCREMENT;
         else if ( m_string == "--" )
            return DECREMENT;
         else if ( m_string == "**" && nextChar != '=' )
            return POW;
         else if ( m_string == "//" )
            m_state = e_eolComment;
         else if ( m_string == "/*" )
            m_state = e_blockComment;
         else if ( m_string == ".=" )
            return FORDOT;
         else if ( parsingFtd() && m_string == "?>" )
         {
            m_mode = t_mOutscape;
            return EOL;
         }
      break;

      case 3:
         if ( m_string == ">>=" )
            return ASSIGN_SHR;
         else if ( m_string == "<<=" )
            return ASSIGN_SHL;
         else if ( m_string == "**=" )
            return ASSIGN_POW;
      break;
   }

   return 0;
}

int SrcLexer::checkLimitedTokens()
{
   switch( m_string.length() )
   {
      case 2:
         if ( m_string == "or" )
            return OR;
         else if ( m_string == "in" )
            return OP_IN;
         else if ( m_string == "if" )
            return IF;
         else if ( m_string == "to" )
            return OP_TO;
      break;

      case 3:
         if ( m_string == "not" )
            return NOT;
         if ( m_string == "try" )
            return TRY;
         else if ( m_string == "nil" )
            return NIL;
         else if ( m_string == "for" )
            return FOR;
         else if ( m_string == "let" )
            return LET;
         else if ( m_string == "has" )
            return HAS;
         else if ( m_string == "and" )
            return AND;
         else if ( m_string == "and" )
            return AND;
         else if ( m_string == "end" )
            return END;
         else if ( m_string == "def" )
            return DEF;

      break;

      case 4:
         if ( m_string == "load" && m_firstSym )  // directive
            return LOAD;
         if ( m_string == "give" )
            return GIVE;
         if ( m_string == "init" )
            return INIT;
         if ( m_string == "else" )
            return ELSE;
         if ( m_string == "elif" )
            return ELIF;
         if ( m_string == "from" )
            return FROM;
         if ( m_string == "self" )
            return SELF;
         if ( m_string == "pass" )
            return PASS;
         if ( m_string == "step" )
            return FOR_STEP;
         if ( m_string == "case" )
            return CASE;
         if ( m_string == "loop" )
            return LOOP;
      break;

      case 5:
         if ( m_string == "catch" )
            return CATCH;
         if ( m_string == "break" )
            return BREAK;
         if ( m_string == "raise" )
            return RAISE;
         if ( m_string == "class" )
            return CLASS;
         if ( m_string == "notin" )
            return OP_NOTIN;
         if ( m_string == "hasnt" )
            return HASNT;
         if ( m_string == "const" )
            return CONST_KW;
         if ( m_string == "while" )
            return WHILE;
      break;

      case 6:
         if ( m_string == "switch" )
            return SWITCH;
         if ( m_string == "select" )
            return SELECT;
         if ( m_string == "global" )
            return GLOBAL;
         if ( m_string == "launch" )
            return LAUNCH;
         if ( m_string == "lambda" )
            return LAMBDA;
         if ( m_string == "sender" )
            return SENDER;
         if ( m_string == "object" )
            return OBJECT;
         if ( m_string == "return" )
            return RETURN;
         if ( m_string == "export" && m_firstSym ) // directive
            return EXPORT;
         if ( m_string == "static" )
            return STATIC;
         if ( m_string == "forall" )
            return FORALL;
      break;

      case 7:
         if ( m_string == "forlast" )
            return FORLAST;
         if ( m_string == "default" )
            return DEFAULT;
      break;


      case 8:
         if ( m_string == "provides" )
            return PROVIDES;
         if ( m_string == "function" )
            return FUNCDECL;
         if ( m_string == "continue" )
            return CONTINUE;
         if ( m_string == "dropping" )
            return DROPPING;
         if ( m_string == "forfirst" )
            return FORFIRST;
      break;

      case 10:
         if ( m_string == "attributes" )
            return ATTRIBUTES;
      break;
   }

   return 0;
}


void SrcLexer::parsingFtd( bool b )
{
   if ( b )
   {
      m_bParsingFtd = true;
      m_mode = t_mOutscape;
   }
   else {
      m_bParsingFtd = false;
      m_mode = t_mNormal;
   }
}

}


/* end of src_lexer.cpp */
