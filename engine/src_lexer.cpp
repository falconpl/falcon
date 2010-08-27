/*
   FALCON - The Falcon Programming Language
   FILE: src_lexer.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 26 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
#include "src_parser.hpp"

namespace Falcon {

SrcLexer::SrcLexer( Compiler *comp ):
   m_value(0),
   m_line( 1 ),
   m_previousLine( 1 ),
   m_character( 0 ),
   m_prevStat(0),
   m_firstEq( false ),
   m_done( false ),
   m_addEol( false ),
   m_lineFilled( false ),
   m_bIsDirectiveLine(false),
   m_incremental( false ),
   m_lineContContext( false ),
   m_graphAgain( false ),
   m_chrEndString(0),
   m_in( 0 ),
   m_compiler( comp ),
   m_state( e_line ),
   m_mode( t_mNormal ),
   m_bParsingFtd(false),
   m_bWasntEmpty(false),
   m_topCtx(0)
{}

SrcLexer::~SrcLexer()
{
   reset();
}

void SrcLexer::reset()
{
   resetContexts();
   m_addEol = false; // revert what's done by resetContext

   m_prevStat = 0;
   m_character = 0;
   m_state = e_line;
   m_bIsDirectiveLine = false;
   m_bWasntEmpty = false;
   m_whiteLead = "";

   while( ! m_streams.empty() )
   {
      Stream *s = (Stream *) m_streams.back();
      m_streams.popBack();
      // all the streams except the first are to be deleted.
      if ( !m_streams.empty() )
         delete s;
   }
}

void SrcLexer::input( Stream *i )
{
   m_in = i;
   m_streams.pushBack(i);
   m_streamLines.pushBack( (uint32) m_line );
   m_done = false;
   m_addEol = false;
   m_lineFilled = false;
}


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
      e_leadIn,
      e_normal,
      e_esc1,
      e_esc2,
      e_escF,
      e_escA,
      e_escL
   }
   state;

   // clear string
   m_string.size( 0 );

   // in outscape mode, everything up to an escape (or at EOF) is a "fast print" statement.
   uint32 chr;
   state = e_leadIn;
   m_state = e_line;
   m_bIsDirectiveLine = false;

   while( m_mode == t_mOutscape && m_in->get( chr ) )
   {
      if ( chr == '\n' )
      {
         m_previousLine = m_line;
         m_line++;
         m_character = 0;
         m_bIsDirectiveLine = false;

         if ( state == e_normal )
            state = e_leadIn;
      }
      else
      {
         m_character++;
      }

      switch( state )
      {
         case e_leadIn:
            if( isWhiteSpace(chr) )
            {
               m_whiteLead.append( chr );
               break;
            }
            else
               state = e_normal;

         // fall through, don't break

         case e_normal:
            if ( chr == '<' )
            {
               state = e_esc1;
            }
            else {
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

               m_string.append( chr );
            }
         break;

         case e_esc1:
            if ( chr == '?' )
               state = e_esc2;
            else {
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

               m_string.append( '<' );
               m_string.append( chr );
               state  = e_normal;
            }
         break;

         case e_esc2:
            if ( chr == '=' )
            {
               // reset lead chars and eventually remove last \n
               m_whiteLead.size(0);
               // but don't remove extra EOL; we want another line.
               m_bWasntEmpty = true;

               // we enter now in the eval mode
               m_mode = t_mEval;
               // and break from the loop so to return the string to print.
               break;
            }
            else if ( isWhiteOrEOL(chr) )
            {

               // reset lead chars and eventually remove last \n
               m_whiteLead.size(0);
               m_bWasntEmpty = m_string.size() > 0;
               if ( m_string.size() > 0 && m_string.getCharAt( m_string.length() -1 ) == '\n' )
                  m_string.size( m_string.size() - m_string.manipulator()->charSize() );

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
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

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
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

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
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

               state = e_normal;
               m_string.append( '<' );
               m_string.append( '?' );
               m_string.append( 'f' );
               m_string.append( 'a' );
               m_string.append( chr );
            }
         break;

         case e_escL:
            if ( isWhiteOrEOL(chr) )
            {
               // reset lead chars and eventually remove last \n
               m_whiteLead.size(0);
               m_bWasntEmpty = m_string.size() > 0;
               if ( m_string.size() > 0 && m_string.getCharAt( m_string.length() -1 ) == '\n' )
                  m_string.size( m_string.size() - m_string.manipulator()->charSize() );

               // we enter now the normal mode; we start to consider a standard program.
               m_mode = t_mNormal;
               // and break from the loop so to return the string to print.
               break;
            }
            else {
               if ( m_whiteLead.size() > 0 )
               {
                  m_string.append( m_whiteLead );
                  m_whiteLead.size(0);
               }

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
      m_string.exported( false );
      return OUTER_STRING;
   }

   // else proceed with normal evaluation as we can't return nothing
   if( m_in->good() )
   {
      switch( m_mode )
      {
         case t_mNormal: return lex_normal();
         case t_mEval: return lex_eval();
         case t_mOutscape: return 0;
      }
   }

   return 0;
}


int SrcLexer::lex_eval()
{
   // prepare for a normal scan
   m_mode = t_mNormal;

   // returns a SHR, which will be interpreted as a fast print
   return SHR;
}


int SrcLexer::lex_normal()
{
   // generate an extra eol?
   if ( m_addEol )
   {
      m_addEol = false;

      // ignore in incremental mode with open contexts
      if ( ! (m_done && incremental() && hasOpenContexts()) )
      {
         m_lineFilled = false;
         if ( ! inParCtx() )
         {
            m_bIsDirectiveLine = false;
            return EOL;
         }
      }
   }

   if( m_graphAgain )
   {
      m_graphAgain = false;
      return CLOSE_GRAPH;
   }

   if ( m_done )
   {
      // raise error if there is some lexer context open
      // in incremental mode ignore completely end of file errors
      if ( ! m_incremental )
         checkContexts();
      return 0;
   }

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
                  {
                     checkContexts();
                     return 0;
                  }

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
      else {
         checkContexts();
         return 0;
      }
   }

   // reset previous token
   m_lineContContext = false;
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
         // if this is the last stream, ask also an extra EOL
         m_streams.popBack();
         if ( m_streams.empty() )
         {
            // fake an empty terminator at the end of input.
            chr = '\n';

            if ( ! m_incremental || ! hasOpenContexts() )
               m_addEol = true;

            m_done = true;
         }
         else {
            delete m_in; // all the streams except first are to be disposed.
            m_in = (Stream *) m_streams.back();
            m_line = (uint32)(int64) m_streamLines.back();
            m_streamLines.popBack();
            m_previousLine = m_line-1;
            next_loop = true;
            continue;
         }
      }

      m_character++;

      // Totally ignore '\r'
      if ( chr == '\r' )
         continue;

      switch ( m_state )
      {
         case e_line:
            // in none status, we have to discard blanks and even ignore '\n';
            // we enter in line or string status depending on what we find.
            if( ! isWhiteSpace( chr ) )
            {
               m_previousLine = m_line;
               if( m_bIsDirectiveLine && chr == '.' )
               {
                  // just ignore it
                  m_string.append( chr );
                  break;
               }

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
               // special x" notation?
               if( m_string.size() == 1 && (chr == '"' || chr == '\''))
               {
                  // recognize only the i" for now
                  if ( m_string[0] == 'i' )
                  {
                     uint32 nextChr;
                     // we'll begin to read a string.
                     if ( readAhead( nextChr ) && nextChr == '\n' )
                     {
                        m_mlString = true;
                        m_line++;
                        m_character = 0;
                        m_in->discardReadAhead( 1 );
                        m_state = chr == '"' ? e_stringRunning : e_litString;
                     }
                     else {
                        m_state = chr == '"' ? e_string : e_litString;
                     }

                     pushContext( ct_string, m_line );
                     m_string.size(0); // remove first char.
                     m_string.exported( true );
                     m_chrEndString = chr; //... up to the matching "
                     break;
                  }
                  // else, just let it through.
               }

               // unless we have a dot in a load directive or namespace
               if( chr == '.' )
               {
                  if ( m_bIsDirectiveLine ||
                     ( m_string.size() != 0 && m_compiler->isNamespace( m_string ))
                  )
                  {
                     // just ignore it
                     m_string.append( chr );
                     break;
                  }
               }


               // push this chr back; we want to read it again in line state
               if( ! isWhiteSpace( chr ) ) // save a loop
                  m_in->unget( chr );

               m_state = e_line;
               // it may be a named token
               int token = checkLimitedTokens();

               if ( token != 0 )
               {
                  return token;
               }
               // internally parsed?
               else if ( m_string.size() == 0 ) {
                  // reset
                  m_state = e_line;
                  continue;
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
               m_in->unget( chr );

               return token;
            }
            else if ( token < 0 || m_string.length() == 3 ) {
               // We have aknowledged this can't be a valid token.
               m_in->unget( chr );
               m_state = e_line;
               m_string = m_string.subString( 0, 1 );
               m_compiler->raiseError( e_inv_token, "'" + m_string + "'", m_line );
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

         case e_eolComment:
            if ( chr == '\n' )
            {
               m_previousLine = m_line;
               m_line ++;
               m_character = 0;
               m_bIsDirectiveLine = false;
               m_state = e_line;

               // a real EOL has been provided here.
               if ( m_state == e_line && ! inParCtx() )
               {
                  m_bIsDirectiveLine = false;
                  if ( m_lineFilled )
                  {
                     m_lineFilled = false;
                     return EOL;
                  }
               }
            }
         break;

         case e_blockComment:
            if ( chr == '\n' )
            { // previous line stays the same
               m_line ++;
               m_character = 0;
               m_bIsDirectiveLine = false;
            }
            else if ( chr == '*' )
            {
               uint32 nextChr;
               readAhead( nextChr );
               if ( nextChr == '/' )
               {
                  m_in->discardReadAhead( nextChr );
                  m_state = e_line;
               }
            }
         break;

         case e_intNumber:
            m_lineFilled = true;
            if ( chr == '.' )
            {
               // a method on a number?
               uint32 nextChr;
               if ( readAhead( nextChr ) && (nextChr < '0' || nextChr > '9') )
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

               // no.
               m_string.append( chr );
               m_state = e_floatNumber;
            }
            else if ( chr == 'e' )
            {
               m_state = e_floatNumber_e;
               m_string.append( chr );
            }
            else if ( chr >= '0' && chr <= '9' )
            {
               m_string.append( chr );
            }
            else if ( chr != '_' )
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
         break;

         case e_floatNumber:
            m_lineFilled = true;
            if ( chr == 'e' )
            {
               m_state = e_floatNumber_e;
               m_string.append( chr );
            }
            else if ( chr >= '0' && chr <= '9'  )
            {
               m_string.append( chr );
            }
            else if ( chr != '_' )
            {
               // end
               m_in->unget( chr );

               // "0." ?
               if( m_string == "0." )
               {
                  m_in->unget( '.' );
                  VALUE->integer = 0;
                  m_state = e_line;
                  return INTNUM;
               }

               numeric retval;
               if ( ! m_string.parseDouble( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->numeric = retval;
               m_state = e_line;
               return DBLNUM;
            }
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
            {
               m_string.size( 0 );
               m_state = e_hexNumber;
            }
            else if ( chr == 'b' || chr == 'B' )
            {
               m_string.size( 0 );
               m_state = e_binNumber;
            }
            else if ( chr == 'c' || chr == 'C' )
            {
               m_string.size( 0 );
               m_state = e_octNumber;
            }
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
            if ( chr >= '0' && chr <= '7' )
            {
               m_string.append( chr );
            }
            else if ( chr != '_' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! m_string.parseOctal( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->integer = retval;
               m_state = e_line;
               return INTNUM;
            }
         break;

         case e_binNumber:
            m_lineFilled = true;
            if ( chr == '0' || chr == '1' )
               m_string.append( chr );
            else if ( chr != '_' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! m_string.parseBin( retval ) )
                  m_compiler->raiseError( e_inv_num_format, m_line );

               VALUE->integer = retval;
               m_state = e_line;
               return INTNUM;
            }
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
            else if ( chr != '_' )
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
            if ( chr == '\n' )
            {
               if( ! m_mlString )
               {
                  m_compiler->raiseError( e_nl_in_lit, m_line );
                  m_lineFilled = false;
                  m_character = 0;
                  m_bIsDirectiveLine = false;
                  m_compiler->raiseError( e_nl_in_lit, m_previousLine );
                  m_state = e_line;
                  popContext();
               }
               m_string.append( chr );
               m_previousLine = m_line;
               m_line++;
            }
            else if ( chr == '\'' )
            {
               uint32 nextChar;
               if ( readAhead( nextChar ) && nextChar == '\'' )
               {
                  m_string.append( '\'' );
                  m_in->discardReadAhead(1);
               }
               else {
                  m_state = e_line;
                  popContext();
                  VALUE->stringp = m_compiler->addString( m_string );
                  m_string.exported( false );
                  return STRING;
               }
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
               readAhead( nextChar );
               switch ( nextChar )
               {
                  case '\\': nextChar = '\\'; break;
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

                  case 'B':
                     nextChar = 1;
                     tempString.size(0);
                     m_state = e_stringBin;
                  break;

                  case '0': case 'c': case 'C':
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
               if( ! m_mlString )
               {
                  m_compiler->raiseError( e_nl_in_lit, m_line );
                  m_lineFilled = false;
                  m_bIsDirectiveLine = false;
                  m_compiler->raiseError( e_nl_in_lit, m_previousLine );
                  m_state = e_line;
                  popContext();
               }
               else
                  m_state = e_stringRunning;

               m_previousLine = m_line;
               m_line++;
               m_character = 0;
               m_bIsDirectiveLine = false;
            }
            else if ( chr == m_chrEndString )
            {
               popContext();
               m_state = e_line;
               VALUE->stringp = m_compiler->addString( m_string );
               m_string.exported( false );
               return STRING;
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
               else if ( chr == m_chrEndString )
               {
                  popContext();
                  m_state = e_line;
                  VALUE->stringp = m_compiler->addString( m_string );
                  m_string.exported( false );
                  return STRING;
               }
               else {
                  m_state = e_string;
                  if ( m_string.size() != 0 )
                     m_string.append( ' ' );
                  m_in->unget( chr );
               }
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
            else if ( chr != '_' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! tempString.parseHex( retval ) || retval > 0xFFFFFFFF )
                  m_compiler->raiseError( e_inv_esc_sequence, m_line );
               m_string.append( (uint32) retval );
               m_state = e_string;
            }
         break;

         case e_stringBin:
            if ( chr == '0' || chr == '1' )
            {
               tempString.append( chr );
            }
            else if ( chr != '_' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! tempString.parseBin( retval ) || retval > 0xFFFFFFFF )
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
            else if ( chr != '_' )
            {
               m_in->unget( chr );
               uint64 retval;
               if ( ! tempString.parseOctal( retval ) || retval > 0xFFFFFFFF )
                  m_compiler->raiseError( e_inv_esc_sequence, m_line );
               m_string.append( (uint32) retval );
               m_state = e_string;
            }
         break;

         default:
            break;
      }
   }

   if ( ! m_incremental )
      checkContexts();

   return 0;
}


void SrcLexer::checkContexts()
{
   t_contextType ct = currentContext();

   if ( ct == ct_round )
      m_compiler->raiseContextError( e_par_unbal, m_line, contextStart() );
   else if ( ct == ct_square )
      m_compiler->raiseContextError( e_square_unbal, m_line, contextStart() );
   else if ( ct == ct_graph )
      m_compiler->raiseContextError( e_graph_unbal, m_line, contextStart() );
   else if ( ct == ct_string  )
      m_compiler->raiseContextError( e_unclosed_string, m_line, contextStart() );
}


int SrcLexer::state_line( uint32 chr )
{
   if ( chr == '\n' )
   {
      m_previousLine = m_line;
      m_line ++;
      m_character = 0;

      // a real EOL has been provided here.
      m_bIsDirectiveLine = false;
      if ( m_lineFilled && ! inParCtx() )
      {
         m_lineFilled = false;
         return EOL;
      }
   }
   else if ( chr == '\\' )
   {
      // don't return at next eol:
      uint32 nextChr;
      if ( ! readAhead( nextChr ) )
      {
         // end of file; if we're in incremental mode,
         // declare the opening of a temporary context
         if ( m_incremental )
         {
            m_lineContContext = true;
         }
      }
      else if ( nextChr == '\n' )
      {
         m_previousLine = m_line;
         m_line ++;
         m_character = 0;
         m_in->discardReadAhead( 1 );
      }
      else if ( nextChr == '\\' )
      {
         m_in->discardReadAhead( 1 );
         parseMacroCall();
      }
      else if ( nextChr == '[' )
      {
         int startline = m_line;
         m_in->discardReadAhead( 1 );
         // create meta data up to \]
         String temp;
         temp.reserve( 512 );
         uint32 chr;
         bool waiting = false;
         while( m_in->get( chr ) )
         {

            if ( chr == '\\' )
            {
               waiting = true;
            }
            else {
               if ( waiting )
               {
                  // done?
                  if ( chr == ']' )
                     break;

                  temp.append( '\\' );
                  waiting = false;
               }

               if ( chr == '\n' )
               {
                  m_previousLine = m_line;
                  m_line ++;
                  m_character = 0;
                  waiting = false;
               }

               temp.append( chr );
            }
         }

         temp.append( '\n' );
         m_compiler->metaCompile( temp, startline );
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
      uint32 nextChr;
      // we'll begin to read a string.
      if ( readAhead( nextChr ) && nextChr == '\n' )
      {
         m_mlString = true;
         m_line++;
         m_character = 0;
         m_in->discardReadAhead( 1 );
         m_state = e_stringRunning;
      }
      else {
         m_mlString = false;
         m_state = e_string;
      }

      pushContext( ct_string, m_line );
      m_chrEndString = '"'; //... up to the matching "
   }
   else if ( chr == 0x201C )
   {
      uint32 nextChr;
      // we'll begin to read a string.
      if ( readAhead( nextChr ) && nextChr == '\n' )
      {
         m_mlString = true;
         m_line++;
         m_character = 1;
         m_in->discardReadAhead( 1 );
      }
      else
         m_mlString = false;
      // we'll begin to read a string.
      m_state = e_string;
      pushContext( ct_string, m_line );
      m_chrEndString = 0x201D; //... up to the matching close quote
   }
   else if ( chr == 0x300C )
   {
      uint32 nextChr;
      // we'll begin to read a string.
      if ( readAhead( nextChr ) && nextChr == '\n' )
      {
         m_mlString = true;
         m_line++;
         m_character = 1;
         m_in->discardReadAhead( 1 );
      }
      else
         m_mlString = false;
      // we'll begin to read a string.
      m_state = e_string;
      pushContext( ct_string, m_line );
      m_chrEndString = 0x300D; //... up to the matching close japanese quote
   }
   else if ( chr == '\'' )
   {
      uint32 nextChr;
      // we'll begin to read a string.
      if ( readAhead( nextChr ) && nextChr == '\n' )
      {
         m_mlString = true;
         m_line++;
         m_character = 1;
         m_in->discardReadAhead( 1 );
      }
      else
         m_mlString = false;

      // we'll begin to read a non escaped string
      pushContext( ct_string, m_line );
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
            m_bIsDirectiveLine = false;
            // but not first sym
            if ( m_lineFilled && ! inParCtx() )
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
         else if ( chr == '&' && nextChar != '=' && nextChar != '&' )
            return AMPER;
         else if ( chr == '~' && nextChar != '=' )
               return TILDE;
         else if ( chr == '|' && nextChar != '=' && nextChar != '|' )
            return VBAR;
         /*
         else if ( chr == '^' && nextChar != '=' && nextChar != '^' )
            return CAP;
         */
         /*else if ( chr == '!' && nextChar != '=' )
               return BANG;
         */
         else if ( chr == '$' )
         {
            return DOLLAR;
         }
         else if ( chr == ':' )
         {
               // but they don't reset first sym
            return COLON;
         }
         else if( chr == ',' )
            return COMMA;
         else if( chr == ';' )
            return ITOK_SEMICOMMA;
         else if ( chr == '.' && nextChar != '=' && nextChar != '[' && nextChar != '"' )
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
            return OP_EQ;
         }
         else if ( chr == '(' || chr == 0xff08 )
         {
            pushContext( ct_round, m_line );
            return OPENPAR;
         }
         else if ( chr == ')' || chr == 0xff09 )
         {
            if ( currentContext() != ct_round )
               m_compiler->raiseError( e_par_close_unbal, m_line );
            else
            {
               popContext();
               return CLOSEPAR;
            }
         }
         else if ( chr == '[' )
         {
            pushContext( ct_square, m_line );
            return OPENSQUARE;
         }
         else if ( chr == ']' )
         {
            if ( currentContext() != ct_square )
            {
               m_compiler->raiseError( e_square_close_unbal, m_line );
            }
            else
            {
               popContext();
               return CLOSESQUARE;
            }
         }
         else if ( chr == '{' )
         {
            pushContext( ct_graph, m_line );
            return OPEN_GRAPH;
         }
         else if ( chr == '}' )
         {
            if ( currentContext() == ct_graph )
            {
               m_graphAgain = true;
               popContext();
               return EOL;
            }
            else
               m_compiler->raiseError( e_graph_close_unbal, m_line );
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
         // EXTENDED CAP OPERATORS
         if ( m_string == "^=" )
            return ASSIGN_BXOR;
         else if ( m_string == "^^" )
            return CAP_CAP;
         else if ( m_string == "^*" )
            return CAP_EVAL;
         else if ( m_string == "^!" )
            return CAP_XOROOB;
         else if ( m_string == "^?" )
            return CAP_ISOOB;
         else if ( m_string == "^-" )
            return CAP_DEOOB;
         else if ( m_string == "^+" )
            return CAP_OOB;
         //====
         else if ( m_string == "=>" )
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
         else if ( m_string == "&&" )
            return AMPER_AMPER;
         else if ( m_string == "||" )
            return VBAR_VBAR;
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
         else if ( m_string == ".[" )
         {
            pushContext( ct_square, m_line );
            return LISTPAR;
         }
         else if ( parsingFtd() && m_string == "?>" && (nextChar != '\n' || m_in->eof() ))
         {
            m_mode = t_mOutscape;
            m_bIsDirectiveLine = false;
            return EOL;
         }
      break;

      case 3:
         if ( parsingFtd() && m_string == "?>\n" )
         {
            m_mode = t_mOutscape;
            if ( m_bWasntEmpty )
               m_whiteLead = "\n";
            m_bIsDirectiveLine = false;
            m_previousLine = m_line;
            m_line++;
            m_character = 0;
            return EOL;
         }
         else if ( m_string == ">>=" )
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
      case 1:
         if ( m_string == "_" )
            return UNB;
      break;

      case 2:
         if ( m_string == "or" )
            return OR;
         else if ( m_string == "in" )
            return OP_IN;
         else if ( m_string == "if" )
            return IF;
         else if ( m_string == "to" )
            return OP_TO;
         else if ( m_string == "as" )
            return OP_AS;
         else if ( m_string == "eq" )
            return OP_EXEQ;
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
         if ( m_string == "load" )  // directive
         {
            m_bIsDirectiveLine = true;
            return LOAD;
         }
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
         if ( m_string == "case" )
            return CASE;
         if ( m_string == "loop" )
            return LOOP;
         if ( m_string == "true" )
            return TRUE_TOKEN;
         if ( m_string == "enum" )
            return ENUM;
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
         if ( m_string == "const" )
            return CONST_KW;
         if ( m_string == "while" )
            return WHILE;
         if ( m_string == "false" )
            return FALSE_TOKEN;
         if ( m_string == "fself" )
            return FSELF;
         if( m_string == "macro" )
         {
            m_string.size(0);
            parseMacro();
            return 0;
         }
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
         if ( m_string == "object" )
            return OBJECT;
         if ( m_string == "return" )
            return RETURN;
         if ( m_string == "export" ) // directive
         {
            m_bIsDirectiveLine = true;
            return EXPORT;
         }
         if ( m_string == "import" ) // directive
         {
            m_bIsDirectiveLine = true;
            return IMPORT;
         }
         if ( m_string == "static" )
            return STATIC;
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

      case 9:
         if ( m_string == "directive" )
         {
            // No assigments in directive.
            m_bIsDirectiveLine = true;
            return DIRECTIVE;
         }
         if ( m_string == "innerfunc" )
            return INNERFUNC;
         if ( m_string == "formiddle" )
            return FORMIDDLE;
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

void SrcLexer::resetContexts()
{
   // clear contexts
   while( m_topCtx != 0 )
   {
      Context* ctx = m_topCtx->m_prev;
      delete m_topCtx;
      m_topCtx = ctx;
   }

   // force to generate a fake eol at next loop
   m_addEol = true;
   m_bIsDirectiveLine = false;

   m_state = e_line;
   m_lineFilled = false;
   m_string = "";
}

void SrcLexer::appendStream( Stream *s )
{
   m_in = s;
   m_streams.pushBack( s );
   m_streamLines.pushBack( (uint32) m_line );
}


void SrcLexer::parseMacro()
{
   // macros are in the form
   // macro decl( params ) (content)
   // they must be passed to the compiler as
   // function decl( params ); > content; end

   int startline = m_line;

   typedef enum {
      s_decl,
      s_params,
      s_endparams,
      s_content,
      s_done
   } macro_state;

   macro_state state = s_decl;
   String sDecl;
   String sContent;
   uint32 ctx = 0;

   uint32 chr;
   while( state != s_done && m_in->get( chr ) )
   {
      if ( chr == '\n' ) {
         m_previousLine = m_line;
         m_line++;
      }

      switch(state) {
         case s_decl:
            if ( chr == '(' )
               state = s_params;
            sDecl += chr;
            break;

         case s_params:
            if ( chr == ')' )
               state = s_endparams;
            sDecl += chr;
            break;

         case s_endparams:
            if ( chr == '(' )
            {
               state = s_content;
               ctx = 1;
            }
            break;

         case s_content:
            if ( chr == '(' )
               ctx++;
            else if( chr == ')' )
            {
               ctx--;
               if ( ctx == 0 )
               {
                  state = s_done;
                  // last ) must not be included
                  break;
               }
            }

            sContent += chr;
            break;

         default:
            break;
      }
   }
   // if we're done, pass the thing to the compiler for metacompilation
   if ( s_done )
   {
      // escape \ and "
      String sContEsc;
      sContent.escape( sContEsc );

      String sFunc = "function " + sDecl + "\n>>@\"" + sContEsc + "\"\nend\n";
      m_compiler->metaCompile( sFunc, startline );
   }
   else {
      // raise an error.
      m_compiler->raiseError( e_syn_macro, sDecl, startline );
   }
}

void SrcLexer::parseMacroCall()
{
   // macros are in the form
   // \\decl( param1, param2 )
   // they must be passed to the compiler as
   // decl( "param1", param2 )

   int startline = m_line;

   typedef enum {
      s_decl,
      s_params,
      s_done
   } macro_state;

   macro_state state = s_decl;
   String sDecl;
   String sParam;
   String sFinal;
   uint32 ctx=0;

   uint32 chr;
   while( state != s_done && m_in->get( chr ) )
   {
      if ( chr == '\n' ) {
         m_previousLine = m_line;
         m_line++;
      }

      switch(state) {
         case s_decl:
            if ( chr == '(' )
            {
               state = s_params;
               ctx = 1;
            }
            sFinal += chr;
            sDecl += chr;
            break;

         case s_params:
            if ( chr == '(' )
            {
               ctx++;
               sParam += chr;
            }
            else if ( chr == ')' )
            {
               ctx--;
               if ( ctx == 0 )
               {
                  state = s_done;
                  if ( sParam.size() > 0 )
                  {
                     String temp;
                     sParam.trim();
                     sParam.escape( temp );
                     sFinal += '"';
                     sFinal += temp;
                     sFinal += '"';
                  }

                  sFinal += chr;
                  break;
               }
               else
                  sParam += chr;
            }
            else if( chr == ',' && ctx == 1 )
            {
               String temp;
               sParam.trim();
               sParam.escape( temp );
               sFinal += '"';
               sFinal += temp;
               sFinal += '"';
               sFinal += ',';
               sParam.size(0);
            }
            else
               sParam += chr;
            break;

         default:
            break;
      }
   }
   // if we're done, pass the thing to the compiler for metacompilation
   if ( s_done )
   {
      m_compiler->metaCompile( sFinal+"\n", startline );
   }
   else {
      // raise an error.
      m_compiler->raiseError( e_syn_macro_call, sDecl, startline );
   }
}


void SrcLexer::pushContext( t_contextType ct, int startLine )
{
   m_topCtx = new Context( ct, startLine, m_topCtx );
}


bool SrcLexer::popContext()
{
   if( m_topCtx == 0 )
      return false;

   Context* current = m_topCtx;
   m_topCtx = m_topCtx->m_prev;
   delete current;
   return true;
}


SrcLexer::t_contextType SrcLexer::currentContext()
{
   if( m_topCtx == 0 )
      return ct_top;

   return m_topCtx->m_ct;
}


int SrcLexer::contextStart()
{
   if( m_topCtx == 0 )
      return 0;

   return m_topCtx->m_oline;
}

bool SrcLexer::inParCtx()
{
   return m_topCtx != 0 &&
         ( m_topCtx->m_ct == ct_round || m_topCtx->m_ct == ct_square );
}

bool SrcLexer::readAhead( uint32 &chr )
{
   bool res;

   while( (res = m_in->readAhead( chr )) && chr == '\r' )
   {
      m_in->discardReadAhead( 1 );
   }

   return res;
}

}


/* end of src_lexer.cpp */
