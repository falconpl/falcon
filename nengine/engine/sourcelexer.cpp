/*
   FALCON - The Falcon Programming Language
   FILE: sourcelexer.cpp

   Lexer detecting the tokens for Falcon script sources
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 18:13:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/textreader.h>
#include <falcon/sourcelexer.h>
#include <falcon/sourcetokens.h>
#include <falcon/trace.h>

#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/tint.h>
#include <falcon/parser/tfloat.h>
#include <falcon/parser/tstring.h>
#include <falcon/parser/tname.h>
#include <falcon/parser/teol.h>
#include <falcon/parser/teof.h>


namespace Falcon {

SourceLexer::SourceLexer( const String& uri, Parsing::Parser* p, TextReader* reader ):
   Parsing::Lexer( uri, p, reader ),
   m_sline( 0 ),
   m_schr( 0 ),
   m_state( state_none ),
   m_nextToken(0)
{
}

SourceLexer::~SourceLexer()
{
   delete m_nextToken;
}

Parsing::TokenInstance* SourceLexer::nextToken()
{
   String tempString;
   char_t chr;

   if ( m_nextToken != 0 )
   {
      Parsing::TokenInstance* inst = m_nextToken;
      m_nextToken = 0;
      return inst;
   }

   while( (chr = m_reader->getChar()) != (char_t)-1 )
   {
      // start the char-line count
      if( m_chr == 0 )
      {
         m_chr = 1;
         m_line = 1;
         if ( chr == '#' )
         {
            m_state = state_shebang1;
         }
      }

      switch( m_state )
      {
         //---- She-bang management
         case state_shebang1:
            if( chr == '!' )
               m_state = state_shebang2;
            else
            {
               m_state = state_line;
               m_text = "#";
            }
            break;

         case state_shebang2:
            if( chr == '\n' ) m_state = state_line;
            break;

         //-------------------------------------------
         //---- normal context -- token recognition.
         case state_none:
            // Ignore '\n' when in state none.
            switch( chr ) {
               case ' ': case '\t': case '\r': case '\n': /* do nothng */ break;
               default:
                  unget(chr);
                  m_state = state_line;
            }
            break;

         case state_line:
            // anything will start from here.
            m_sline = m_line;
            m_schr = m_chr;
            m_text.size(0);

            switch(chr) {
               case ' ': case '\t': case '\r': /* do nothng */ break;
               case '\n':
                  {
                  int32 l = m_line;
                  int32 c = m_chr;
                  m_line++;
                  m_chr = 1;
                  // After a real new-line, enter in none-state
                  m_state = state_none;
                  return Parsing::t_eol().makeInstance(l, c);
                  }

               case ';': return Parsing::t_eol().makeInstance(m_line, m_chr++);
               case '"': m_state = state_double_string; break;
               case '\'': m_state = state_single_string; break;
               case '0': m_state = state_zero_prefix; break;

               default:
                  if ( isCipher(chr) )
                  {
                     m_text.append(chr);
                     m_state = state_integer;
                  }
                  else if (!isTokenLimit(chr))
                  {
                     m_text.append(chr);
                     m_state = state_name;
                  }
                  else
                  {
                     m_text.append( chr );
                     m_state = state_operator;
                  }
            }
            break;

         //----------------------------------------------
         // Comment context

         case state_eolComment:
            if ( chr == '\n' )
            {
               m_state = state_line;
               int32 l = m_line;
               int32 c = m_chr;
               m_line++;
               m_chr = 1;
               return Parsing::t_eol().makeInstance(l, c);
            }
            break;

         case state_blockComment:
            switch( chr )
            {
               case '*': m_state = state_blockCommentAsterisk; break;
               case '\\': m_state = state_blockCommentEscape; break;
            }
            break;

         case state_blockCommentAsterisk:
            switch( chr )
            {
               case '*': /* do nothing */; break;
               case '/': m_state = state_blockCommentEscape; break;
               default: m_state = state_blockComment; break;
            }
            break;

         case state_blockCommentEscape:
            m_state = state_blockComment;
            switch( chr )
            {
               case '*': /* do nothing */; break;
               case '/': m_state = state_blockCommentEscape; break;
               default: ; break;
            }
            break;

         //---------------------------------------
         //---- string context

         case state_double_string:
            switch( chr ) {
               case '\"': return Parsing::t_string().makeInstance( m_sline, m_schr, m_text );
               case '\\': m_state = state_double_string_esc; break;
               case '\n': m_state = state_double_string_nl; break;
               default:
                  m_text.append(chr);
            }
            break;


         case state_double_string_nl:
            switch( chr )
            {
               case ' ': case '\n': case '\r': case '\t': /* do nothing */ break;
               case '\\': m_state = state_double_string_esc; break;
               default:
                  m_text.append(chr);
                  m_state = state_double_string;
            }
            break;

         case state_double_string_esc:
            switch( chr )
            {
               case '\\': m_text.append('\\'); m_state = state_double_string; break;
               case 'n': m_text.append('\n'); m_state = state_double_string; break;
               case 'b': m_text.append('\b'); m_state = state_double_string; break;
               case 't': m_text.append('\t'); m_state = state_double_string; break;
               case 'r': m_text.append('\r'); m_state = state_double_string; break;
               case 'x': case 'X': m_state = state_double_string_hex; break;
               case 'B': m_state = state_double_string_bin; break;
               case '0': case 'c': case 'C': m_state = state_double_string_octal; break;
               default:
                  m_state = state_double_string;
                  addError(e_inv_esc_sequence);
            }
            break;

         case state_double_string_hex:
            if (  (chr >= '0' && chr <= '9') ||
                     (chr >= 'a' && chr <= 'f') ||
                     (chr >= 'A' && chr <= 'F')
               )
            {
               tempString.append( chr );
            }
            else if ( chr != '_' )
            {
               m_reader->ungetChar( chr );
               uint64 retval;
               if ( ! tempString.parseHex( retval ) || retval > 0xFFFFFFFF )
               {
                  addError( e_inv_esc_sequence );
               }
               m_text.append( (char_t) retval );
               m_state = state_double_string;
               continue; // skip char advancement control
            }
            break;

         case state_double_string_bin:
            if ( chr == '0' || chr == '1' )
            {
               tempString.append( chr );
            }
            else if ( chr != '_' )
            {
               m_reader->ungetChar( chr );
               uint64 retval;
               if ( ! tempString.parseBin( retval ) || retval > 0xFFFFFFFF )
               {
                  addError( e_inv_esc_sequence );
               }
               m_text.append( (char_t) retval );
               m_state = state_double_string;
               continue; // skip char advancement control
            }
         break;

         case state_double_string_octal:
            if (  (chr >= '0' && chr <= '7') )
            {
               tempString.append( chr );
            }
            else if ( chr != '_' )
            {
               m_reader->ungetChar( chr );
               uint64 retval;
               if ( ! tempString.parseOctal( retval ) || retval > 0xFFFFFFFF )
               {
                  addError( e_inv_esc_sequence );
               }
               m_text.append( (char_t) retval );
               m_state = state_double_string;
               continue; // skip char advancement control
            }
         break;


         case state_single_string:
            if( chr == '\'')
            {
               char_t chr1;
               // do a read ahead
               if( ( chr1 = m_reader->getChar()) != (char_t) -1 )
               {
                  if ( chr1 == '\'')
                  {
                     m_text.append( '\'' );
                  }
                  else
                  {
                     m_reader->ungetChar(chr1);
                     m_state = state_line;
                     return Parsing::t_string().makeInstance( m_sline, m_schr, m_text );
                  }
               }
               // on read failure, will break at next loop
            }
            else
            {
               m_text.append(chr);
            }
            break;

         //---------------------------------------
         //---- numnber context

         case state_zero_prefix:
            switch( chr )
            {
               case 'x': case 'X': m_state = state_hex; break;
               case 'b': case 'B': m_state = state_bin; break;
               case 'c': case 'C':m_state = state_octal; break;
               case '.': m_text += "0."; m_state = state_float_first; break;
               default:
                  if ( chr >= '0' && chr <= '7' )
                  {
                     m_text.append( chr );
                     m_state = state_octal;
                  }
                  else
                  {
                     m_state = state_line;

                     if ( isTokenLimit(chr) )
                     {
                        unget( chr );
                        return Parsing::t_int().makeInstance(m_sline, m_schr, 0);
                     }
                     addError(e_inv_num_format);
                  }
            }
            break;

         case state_octal:
            if ( chr >= '0' && chr <= '7' )
            {
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseOctal( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_int().makeInstance(m_sline, m_schr, (int64) retval);
            }
         break;

         case state_bin:
            if ( chr >= '0' && chr <= '1' )
            {
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseBin( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_int().makeInstance(m_sline, m_schr, (int64) retval);
            }
         break;

         case state_hex:
            if (  (chr >= '0' && chr <= '9') ||
                  (chr >= 'a' && chr <= 'f') ||
                  (chr >= 'A' && chr <= 'F')
               )
            {
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseHex( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_int().makeInstance(m_sline, m_schr, (int64) retval);
            }
         break;

         case state_integer:
            if (  (chr >= '0' && chr <= '9') )
            {
               m_text.append( chr );
            }
            else if ( chr == '.' )
            {
               m_text.append( '.' );
               m_state = state_float_first;
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               int64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseInt( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_int().makeInstance(m_sline, m_schr, retval);
            }
         break;


         case state_float_first:
            if (  (chr >= '0' && chr <= '9') )
            {
               m_text.append( chr );
               m_state = state_float;
            }
            else if ( isSymStart(chr) )
            {
               m_state = state_line;
               unget( chr );
               int64 retval = 1; // to avoid stupid division by zero in case of errors
               m_text.remove(m_text.length()-1,1);
               if ( ! m_text.parseInt( retval ) )
                  addError( e_inv_num_format );

               m_nextToken = t_dot().makeInstance(m_sline, m_schr);
               return Parsing::t_int().makeInstance(m_sline, m_schr, retval);
            }
            else if ( chr != '_' )
            {
              addError( e_inv_num_format );
              return Parsing::t_int().makeInstance(m_sline, m_schr, 1);
            }
            break;

         case state_float:
            if (  (chr >= '0' && chr <= '9') )
            {
               m_state = state_float;
               m_text.append( chr );
            }
            else if ( chr == 'e' || chr == 'E' )
            {
               m_state = state_float_exp;
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               double retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseDouble( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_float().makeInstance(m_sline, m_schr, retval);
            }
            break;

         case state_float_exp:
            if (  (chr >= '0' && chr <= '9') )
            {
               m_state = state_float;
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               m_state = state_line;
               unget( chr );
               double retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseDouble( retval ) )
                  addError( e_inv_num_format );

               return Parsing::t_float().makeInstance(m_sline, m_schr, retval);
            }
            break;

         //---------------------------------------
         //---- language tokens or symbols

         case state_name:
            if (isTokenLimit(chr))
            {
               unget(chr);
               m_state = state_line;
               return checkWord();
            }
            m_text.append( chr );
            break;

         case state_operator:
            if( String::isWhiteSpace( chr ) || !isTokenLimit( chr ) )
            {
               unget(chr);
               m_state = state_line;
               return checkOperator();
            }
            break;
      }

      //--------------------------------------------------------------
      // Character +  line advance
      // Do this now so we can advance only after having read a char.
      if( chr == '\n' )
      {
         TRACE2( "SourceLexer::nextToken - at line %d", m_line );
         m_line++;
         m_chr = 0;
      }
      m_chr++;
   }

   return 0;
}


Parsing::TokenInstance* SourceLexer::checkWord()
{
   switch(m_text.length())
   {
      case 1:
         //if ( m_text == "_" ) return UNB;
      break;

      case 2:
         if ( m_text == "as" ) return t_token_as().makeInstance(m_sline, m_schr);
         if ( m_text == "eq" ) return t_token_eq().makeInstance(m_sline, m_schr);
         if ( m_text == "in" ) return t_token_in().makeInstance(m_sline, m_schr);
         if ( m_text == "if" ) return t_token_if().makeInstance(m_sline, m_schr);
         if ( m_text == "or" ) return t_token_or().makeInstance(m_sline, m_schr);
         if ( m_text == "to" ) return t_token_to().makeInstance(m_sline, m_schr);
      break;

      case 3:
         if ( m_text == "not" ) return t_token_not().makeInstance(m_sline, m_schr);
         if ( m_text == "end" ) return t_token_end().makeInstance(m_sline, m_schr);
         if ( m_text == "nil" ) return t_token_nil().makeInstance(m_sline, m_schr);
         /*else if ( m_text == "try" )
         else if ( m_text == "for" )
         else if ( m_text == "and" )
         else if ( m_text == "and" )
         else if ( m_text == "def" )
         */
      break;

      case 4:
         /*
         if ( m_text == "load" )  // directive
         {
            m_bIsDirectiveLine = true;
            return LOAD;
         }
         if ( m_text == "init" )
            return INIT;
         if ( m_text == "else" )
            return ELSE;
         if ( m_text == "elif" )
            return ELIF;
         if ( m_text == "from" )
            return FROM;
         if ( m_text == "self" )
            return SELF;
         if ( m_text == "case" )
            return CASE;
         if ( m_text == "loop" )
            return LOOP;
         if ( m_text == "true" )
            return TRUE_TOKEN;
         if ( m_text == "enum" )
            return ENUM;
          */
      break;

      case 5:
         /*
         if ( m_text == "catch" )
            return CATCH;
         if ( m_text == "break" )
            return BREAK;
         if ( m_text == "raise" )
            return RAISE;
         if ( m_text == "class" )
            return CLASS;
         if ( m_text == "notin" )
            return OP_NOTIN;
         if ( m_text == "const" )
            return CONST_KW;
         if ( m_text == "while" )
            return WHILE;
         if ( m_text == "false" )
            return FALSE_TOKEN;
         if ( m_text == "fself" )
            return FSELF;
         if( m_text == "macro" )
         {
            m_text.size(0);
            parseMacro();
            return 0;
         }
          */
      break;

      case 6:
         /*
         if ( m_text == "switch" )
            return SWITCH;
         if ( m_text == "select" )
            return SELECT;
         if ( m_text == "global" )
            return GLOBAL;
         if ( m_text == "launch" )
            return LAUNCH;
         if ( m_text == "object" )
            return OBJECT;
         if ( m_text == "return" )
            return RETURN;
         if ( m_text == "export" ) // directive
         {
            m_bIsDirectiveLine = true;
            return EXPORT;
         }
         if ( m_text == "import" ) // directive
         {
            m_bIsDirectiveLine = true;
            return IMPORT;
         }
         if ( m_text == "static" )
            return STATIC;
         */
      break;

      case 7:
         /*
         if ( m_text == "forlast" )
            return FORLAST;
         if ( m_text == "default" )
            return DEFAULT;
         */
      break;


      case 8:
         /*
         if ( m_text == "provides" )
            return PROVIDES;
         if ( m_text == "function" )
            return FUNCDECL;
         if ( m_text == "continue" )
            return CONTINUE;
         if ( m_text == "dropping" )
            return DROPPING;
         if ( m_text == "forfirst" )
            return FORFIRST;
          */
      break;

      case 9:
         /*
         if ( m_text == "directive" )
         {
            // No assigments in directive.
            m_bIsDirectiveLine = true;
            return DIRECTIVE;
         }
         if ( m_text == "innerfunc" )
            return INNERFUNC;
         if ( m_text == "formiddle" )
            return FORMIDDLE;
          */
      break;
   }

   // As a fallback, create a "name" word
   return Parsing::t_name().makeInstance( m_sline, m_schr, m_text );
}


Parsing::TokenInstance* SourceLexer::checkOperator()
{
   switch(m_text.length())
   {
      case 1:
         if( m_text == "+" ) return t_plus().makeInstance(m_sline, m_schr);
         else if( m_text == "-" ) return t_minus().makeInstance(m_sline, m_schr);
         else if( m_text == "*" ) return t_times().makeInstance(m_sline, m_schr);
         else if( m_text == "/" ) return t_divide().makeInstance(m_sline, m_schr);
         else if( m_text == "%" ) return t_modulo().makeInstance(m_sline, m_schr);
         break;

      case 2:
         if( m_text == "**" ) return t_pow().makeInstance(m_sline, m_schr);
         break;
   }
   // in case of error
   addError( e_inv_token );
   // Create an unary operator -- pretty almost always ok.
   return t_token_not().makeInstance(m_sline, m_schr);
}

}

/* end of sourcelexer.cpp */
