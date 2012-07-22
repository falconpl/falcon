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

#undef SRC
#define SRC "engine/sp/sourcelexer.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/textreader.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/parser/parser.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/trace.h>

#include <falcon/parser/tokeninstance.h>

#include <set>

namespace Falcon {

class SourceLexer::Private
{
public:
   typedef std::set<String> StringSet;
   StringSet m_nsSet;
};


SourceLexer::SourceLexer( const String& uri, Parsing::Parser* p, TextReader* reader ):
   Parsing::Lexer( uri, p, reader ),
   _p( new Private ),
   m_sline( 0 ),
   m_schr( 0 ),
   m_hadOperator( false ),
   m_hadImport(false),
   m_stringStart( false ),
   m_stringML( false ),
   m_state( state_none ),
   m_nextToken(0)
{
}

SourceLexer::~SourceLexer()
{
   if( m_nextToken != 0 ) m_nextToken->dispose();
   delete _p;
}

Parsing::TokenInstance* SourceLexer::nextToken()
{
   SourceParser* parser = static_cast<SourceParser*>(m_parser);
   String tempString;
   char_t chr;
   t_state previousState = state_none;
   int curMemChr = 0;

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
               resetState();
               m_text = "#";
            }
            break;

         case state_shebang2:
            if( chr == '\n' ) resetState();
            break;

         //-------------------------------------------
         //---- normal context -- token recognition.
         case state_none:
            // Ignore '\n' when in state none.
            switch( chr ) {
               case ' ': case '\t': case '\r': case '\n': /* do nothng */ break;
               case '/': previousState = m_state; m_state = state_enterComment; break;
               default:
                  unget(chr);
                  resetState();
                  //m_hadOperator = true;
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
                  // return only if not an operator at end of line.
                  if( ! m_hadOperator )
                  {
                     return m_parser->T_EOL.makeInstance(l, c);
                  }
                  break;
               }
               
               case ';': m_hadOperator = true; return m_parser->T_EOL.makeInstance(m_line, m_chr++);
               case ':': m_hadOperator = true; return parser->T_Colon.makeInstance(m_line, m_chr++);
               case ',': m_hadOperator = true; return parser->T_Comma.makeInstance(m_line, m_chr++);
               case '"':  m_stringML = false; m_stringStart = true; m_state = state_double_string; break;
               case '\'': m_stringML = false; m_stringStart = true; m_state = state_single_string; break;
               case '0': m_state = state_zero_prefix; break;
               case '(': m_chr++; m_hadOperator = true; return parser->T_Openpar.makeInstance(m_line,m_chr); break;
               case ')': m_chr++; resetState(); return parser->T_Closepar.makeInstance(m_line,m_chr); break;
               case '[': m_chr++; m_hadOperator = true; return parser->T_OpenSquare.makeInstance(m_line,m_chr); break;
               case ']': m_chr++; resetState(); return parser->T_CloseSquare.makeInstance(m_line,m_chr); break;
               case '{': m_chr++; m_hadOperator = true; return parser->T_OpenGraph.makeInstance(m_line,m_chr); break;
               case '}': m_chr++;
                  resetState();
                  m_nextToken = parser->T_end.makeInstance(m_line,m_chr);
                  return parser->T_EOL.makeInstance(m_line,m_chr);

               case '/': previousState = m_state; m_state = state_enterComment; break;

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

         case state_enterComment:
            switch( chr )
            {
               case '/': m_state = state_eolComment; break;
               case '*': m_state = state_blockComment; break;
               default:
                  // it was a simple '/'
                  m_text.size(0); m_text.append('/');
                  unget(chr);
                  // sline might be ok, but not if we're here from state none.
                  // neutralize next character
                  m_chr--; chr = ' ';
                  m_sline = m_line;
                  m_schr = chr; // where '/' was opened.
                  // the state is operator (probably a division).
                  m_state = state_operator;
            }

         case state_eolComment:
            if ( chr == '\n' )
            {
               unget(chr);
               m_state = previousState;
               // to allow correct processing of next loop -- \n will fix line/chr
               chr = ' ';
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
               case '/': m_state = previousState; break;
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
               case '"': m_chr++; resetState(); return m_parser->T_String.makeInstance( m_sline, m_schr, m_text );
               case '\\': m_state = state_double_string_esc; break;

               case '\n':
                  if( m_stringStart )
                  {
                     m_stringML = true;
                  }
                  else
                  {
                     if ( m_stringML )
                     {
                        m_text.append(' ');
                     }
                     else
                     {
                        addError(e_nl_in_lit);
                        m_stringML = true; // to avoid multiple error reports.
                     }
                  }
                  m_state = state_double_string_nl; break;

               default:
                  m_text.append(chr);
            }
            m_stringStart = false;
            break;


         case state_double_string_nl:
            switch( chr )
            {
               case ' ': case '\n': case '\r': case '\t': /* do nothing */ break;
               case '\\': m_state = state_double_string_esc; break;
               case '"':
                  m_chr++;
                  resetState();
                  return m_parser->T_String.makeInstance( m_sline, m_schr, m_text );

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
                     resetState();
                     return m_parser->T_String.makeInstance( m_sline, m_schr, m_text );
                  }
               }
               // on read failure, will break at next loop
            }
            else
            {
               if ( chr == '\n' )
               {
                  // at string start? -- allow it, but don't record it.
                  if ( m_stringStart )
                  {
                     m_stringML = true;
                  }
                  else if( m_stringML )
                  {
                     m_text.append('\n');
                  }
                  else
                  {
                     addError(e_nl_in_lit);
                     // however, add it.
                     m_text.append('\n');
                     m_stringML = true; // to avoid multiple error reports.
                  }
               }
               else {
                  m_text.append(chr);
               }
            }
            // never at start after first char.
            m_stringStart = false;
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
                     resetState();

                     if ( isTokenLimit(chr) )
                     {
                        unget( chr );
                        return m_parser->T_Int.makeInstance(m_sline, m_schr, 0);
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
               resetState();
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseOctal( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Int.makeInstance(m_sline, m_schr, (int64) retval);
            }
         break;

         case state_bin:
            if ( chr >= '0' && chr <= '1' )
            {
               m_text.append( chr );
            }
            else if ( chr != '_' )
            {
               resetState();
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseBin( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Int.makeInstance(m_sline, m_schr, (int64) retval);
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
               resetState();
               unget( chr );
               uint64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseHex( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Int.makeInstance(m_sline, m_schr, (int64) retval);
            }
         break;


         case state_membuf:
            if ( chr >= '0' && chr <= '9' )
            {
               curMemChr = chr - '0';
            }
            else if (chr >= 'a' && chr <= 'f') {
               curMemChr = chr - 'a' + 10;
            }
            else if(chr >= 'A' && chr <= 'F' ) {
               curMemChr = chr - 'A' + 10;
            }
            else if( chr == '}' ) {
               // we're done
               m_chr++;
               resetState();
               Parsing::TokenInstance* ti = m_parser->T_String.makeInstance( m_sline, m_schr, m_text );
               ti->asString()->toMemBuf();
               return ti;
            }
            else if(chr == '\n' ) {
               // ignore \n, but add newline.
               m_line++;
               m_chr = 0;
               break;
            }
            else if( chr == '_' || chr == ' ' || chr =='\r' || chr =='\t' ) {
               // ignore whitespaces.
               m_chr++;
               break;
            }
            else {
               m_parser->addError(e_membuf_def, m_parser->currentSource(), m_line, m_chr, 0);
               // proceed till '}'
               m_state = state_membuf3;
               break;
            }

            // normally, change state
            m_state = state_membuf2;
            m_chr++;
         break;

         case state_membuf2:
            if ( chr >= '0' && chr <= '9' )
            {
               curMemChr = curMemChr << 4 | (chr - '0');
            }
            else if (chr >= 'a' && chr <= 'f') {
               curMemChr = curMemChr << 4 | (chr - 'a' + 10);
            }
            else if(chr >= 'A' && chr <= 'F' ) {
               curMemChr = curMemChr << 4 | (chr - 'A' + 10);
            }
            else if(chr == '\n' ) {
               m_parser->addError(e_membuf_def, m_parser->currentSource(), m_line, m_chr, 0);
               m_state = state_membuf3;
               m_line++;
               m_chr = 0;
            }
            else {
               m_parser->addError(e_membuf_def, m_parser->currentSource(), m_line, m_chr, 0);
               // proceed till '}'
               m_state = state_membuf3;
            }
            m_text.append( curMemChr );
            curMemChr = 0;
            m_state = state_membuf;
         break;

         // consume up to }
         case state_membuf3:
            if(chr == '\n' ) {
               m_line++;
               m_chr = 0;
            }
            else if( chr == '}' ) {
               m_chr++;
               resetState();
               Parsing::TokenInstance* ti = m_parser->T_String.makeInstance( m_sline, m_schr, m_text );
               ti->asString()->toMemBuf();
               return ti;
            }
            else {
               m_chr++;
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
               resetState();
               unget( chr );
               int64 retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseInt( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Int.makeInstance(m_sline, m_schr, retval);
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
               resetState();
               unget( chr );
               double retval = 1; // to avoid stupid division by zero in case of errors
               m_text.remove(m_text.length()-1,1);
               if ( ! m_text.parseDouble( retval ) )
                  addError( e_inv_num_format );

               m_nextToken = parser->T_Dot.makeInstance(m_sline, m_schr);
               return m_parser->T_Float.makeInstance(m_sline, m_schr, retval);
            }
            else if ( chr != '_' )
            {
              addError( e_inv_num_format );
              m_chr++;
              resetState();
              return m_parser->T_Int.makeInstance(m_sline, m_schr, 1);
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
               resetState();
               unget( chr );
               double retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseDouble( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Float.makeInstance(m_sline, m_schr, retval);
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
               resetState();
               unget( chr );
               double retval = 1; // to avoid stupid division by zero in case of errors
               if ( ! m_text.parseDouble( retval ) )
                  addError( e_inv_num_format );

               return m_parser->T_Float.makeInstance(m_sline, m_schr, retval);
            }
            break;

         //---------------------------------------
         //---- language tokens or symbols

         case state_name:
            if (isTokenLimit(chr))
            {
               // special cases
               if( m_text == "p" && chr == '{' )
               {
                  m_chr++;
                  resetState();
                  m_text.size(0);
                  return parser->T_OpenProto.makeInstance( m_sline, m_schr );
               }
               else if( m_text == "m" && chr == '{' )
               {
                  m_chr++;
                  m_state = state_membuf;
                  m_text.size(0);
                  break;
               }
               
               // namespace check
               else if( chr != '.' || ! isNameSpace( m_text ) )
               {
                  unget(chr);
                  resetState();
                  return checkWord();
               }
            }
            m_text.append( chr );
            break;

         case state_operator:
            if( String::isWhiteSpace( chr ) ||
               isParenthesis(chr) || chr == '\'' || chr == '"' 
               || chr == '$' || chr == '#'
               || !isTokenLimit( chr ) )
            {
               // special case -- dot/square
               if( m_text == ".")
               {
                  if( chr == '[' )
                  {
                     m_chr++;
                     resetState();
                     m_hadOperator = true;
                     return parser->T_DotSquare.makeInstance(m_sline, m_schr );
                  }
                  else if ( chr == '(' )
                  {
                     m_chr++;
                     resetState();
                     m_hadOperator = true;
                     return parser->T_DotPar.makeInstance(m_sline, m_schr );
                  }
               }               

               unget(chr);
               // reset the state, but don't ignore previous had-operator
               bool b = m_hadOperator;
               resetState();
               m_hadOperator = b;

               Parsing::TokenInstance* ti = checkOperator();
               if( ti != 0 )
               {
                  return ti;
               }
               // else we had an error; try to go on.
            }
            m_text.append( chr );
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

   if ( m_state == state_line )
   {
      m_state = state_none;
      // generate an extra EOL if we had an open line.
      return m_parser->T_EOL.makeInstance(m_line, m_chr);
   }
   else if ( m_state != state_none )
   {
      // if state is none, we don't need to generate an extra EOL, but if
      // state is anything else, then we have an open context error.
      m_parser->addError( e_syntax, m_uri, m_line, m_chr, m_sline );
   }

   return 0;
}


void SourceLexer::resetState()
{
   m_state = state_line;
   m_hadOperator = false; // checkOperator will set it back as needed.
   m_hadImport = false;
}

Parsing::TokenInstance* SourceLexer::checkWord()
{
   SourceParser* parser = static_cast<SourceParser*>(m_parser);

   switch(m_text.length())
   {
      case 1:
         //if ( m_text == "_" ) return UNB;
      break;

      case 2:
         if ( m_text == "as" ) return parser->T_as.makeInstance(m_sline, m_schr);
         if ( m_text == "in" ) return parser->T_in.makeInstance(m_sline, m_schr);
         if ( m_text == "if" ) return parser->T_if.makeInstance(m_sline, m_schr);
         if ( m_text == "or" ) return parser->T_or.makeInstance(m_sline, m_schr);
         if ( m_text == "to" ) return parser->T_to.makeInstance(m_sline, m_schr);
      break;

      case 3:
         if ( m_text == "and" ) return parser->T_and.makeInstance(m_sline, m_schr);
         if ( m_text == "def" ) return parser->T_def.makeInstance(m_sline, m_schr);
         if ( m_text == "end" ) return parser->T_end.makeInstance(m_sline, m_schr);
         if ( m_text == "for" ) return parser->T_for.makeInstance(m_sline, m_schr);
         if ( m_text == "nil" ) return parser->T_nil.makeInstance(m_sline, m_schr);
         if ( m_text == "not" ) return parser->T_not.makeInstance(m_sline, m_schr);
         if ( m_text == "try" ) return parser->T_try.makeInstance(m_sline, m_schr);
      break;

      case 4:
         if ( m_text == "elif" ) return parser->T_elif.makeInstance(m_sline, m_schr);
         if ( m_text == "else" ) return parser->T_else.makeInstance(m_sline, m_schr);
         if ( m_text == "init" ) return parser->T_init.makeInstance(m_sline, m_schr);
         if ( m_text == "rule" ) return parser->T_rule.makeInstance(m_sline, m_schr);
         if ( m_text == "self" ) return parser->T_self.makeInstance(m_sline, m_schr);
         if ( m_text == "true" ) return parser->T_true.makeInstance(m_sline, m_schr);
         if ( m_text == "from" ) return parser->T_from.makeInstance(m_sline, m_schr);
         if ( m_text == "load" ) return parser->T_load.makeInstance(m_sline, m_schr);
         if ( m_text == "case" ) return parser->T_case.makeInstance(m_sline, m_schr);
         
         /*
         if ( m_text == "loop" )
            return LOOP;
         if ( m_text == "enum" )
            return ENUM;
          */
      break;

      case 5:
          if ( m_text == "while" ) return parser->T_while.makeInstance(m_sline, m_schr);
          if ( m_text == "false" ) return parser->T_false.makeInstance(m_sline, m_schr);
          if ( m_text == "class" ) return parser->T_class.makeInstance(m_sline, m_schr);
          if ( m_text == "break" ) return parser->T_break.makeInstance(m_sline, m_schr);
          if ( m_text == "notin" ) return parser->T_notin.makeInstance(m_sline, m_schr);
          if ( m_text == "catch" ) return parser->T_catch.makeInstance(m_sline, m_schr);
          if ( m_text == "raise" ) return parser->T_raise.makeInstance(m_sline, m_schr);
          if ( m_text == "fself" ) return parser->T_fself.makeInstance(m_sline, m_schr);
          
          /*
         if ( m_text == "const" )
            return CONST_KW;
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
         if ( m_text == "global" )
            return GLOBAL;
         if ( m_text == "launch" )
            return LAUNCH;
         if ( m_text == "object" )
            return OBJECT;
            */
         if ( m_text == "switch" ) return parser->T_switch.makeInstance(m_sline, m_schr);
         if ( m_text == "select" ) return parser->T_select.makeInstance(m_sline, m_schr);
         if ( m_text == "return" ) return parser->T_return.makeInstance(m_sline, m_schr);
         if ( m_text == "export" ) return parser->T_export.makeInstance(m_sline, m_schr);
         if ( m_text == "import" ) 
         {
            m_hadImport = true;
            return parser->T_import.makeInstance(m_sline, m_schr);
         }
         
         /*
         if ( m_text == "static" )
            return STATIC;
         */
      break;

      case 7:
         if ( m_text == "forlast" ) return parser->T_forlast.makeInstance(m_sline, m_schr);
         if ( m_text == "finally" ) return parser->T_finally.makeInstance(m_sline, m_schr);
         if ( m_text == "default" ) return parser->T_default.makeInstance(m_sline, m_schr);
      break;


      case 8:
         /*
         if ( m_text == "provides" )
            return PROVIDES;
           */
         if ( m_text == "forfirst" ) return parser->T_forfirst.makeInstance(m_sline, m_schr);
         
         if ( m_text == "function" )
            return parser->T_function.makeInstance(m_sline, m_schr);
         
         if ( m_text == "continue" )
            return parser->T_continue.makeInstance(m_sline, m_schr);
      break;

      case 9:
         if ( m_text == "namespace" )
            return parser->T_namespace.makeInstance(m_sline, m_schr);         
         if ( m_text == "formiddle" ) return parser->T_formiddle.makeInstance(m_sline, m_schr);
        /*
         if ( m_text == "directive" )
         {
            // No assigments in directive.
            m_bIsDirectiveLine = true;
            return DIRECTIVE;
         }
         */
      break;
   }

   // As a fallback, create a "name" word
   if( !m_hadImport && isNameSpace(m_text) )
   {
      m_parser->addError( e_ns_clash, m_parser->currentSource(), m_sline, m_schr, 0, m_text );
   }
   return m_parser->T_Name.makeInstance( m_sline, m_schr, m_text );
}


Parsing::TokenInstance* SourceLexer::checkOperator()
{
   SourceParser* parser = static_cast<SourceParser*>(m_parser);

   bool bOp = m_hadOperator || m_state == state_none;
   m_hadOperator = true;
   switch(m_text.length())
   {
      case 1:
         if( m_text == "+" ) return parser->T_Plus.makeInstance(m_sline, m_schr);
         if( m_text == "*" ) return parser->T_Times.makeInstance(m_sline, m_schr);
         if( m_text == "/" ) return parser->T_Divide.makeInstance(m_sline, m_schr);
         if( m_text == "%" ) return parser->T_Modulo.makeInstance(m_sline, m_schr);
         if( m_text == "=" ) return parser->T_EqSign.makeInstance(m_sline, m_schr);
         if( m_text == "<" ) return parser->T_Less.makeInstance(m_sline, m_schr);
         if( m_text == ">" ) return parser->T_Greater.makeInstance(m_sline, m_schr);
         if( m_text == "." ) return parser->T_Dot.makeInstance(m_sline, m_schr);
         if( m_text == ":" ) return parser->T_Colon.makeInstance(m_sline, m_schr);
         if( m_text == "," ) return parser->T_Comma.makeInstance(m_sline, m_schr);
         if( m_text == "$" ) return parser->T_Dollar.makeInstance(m_sline, m_schr);
         if( m_text == "&" ) return parser->T_Amper.makeInstance(m_sline, m_schr);
         if( m_text == "?" ) return parser->T_QMark.makeInstance(m_sline, m_schr);
         // the cut operator is a statement.
         if( m_text == "!" )
         {
            m_hadOperator = false;
            return parser->T_Bang.makeInstance(m_sline, m_schr);
         }

         if( m_text == "-" )
            return bOp ?
               parser->T_UnaryMinus.makeInstance( m_sline, m_schr):
               parser->T_Minus.makeInstance(m_sline, m_schr);
         break;

      case 2:
         if( m_text == "++" )
         {
            m_hadOperator = false;
            return parser->T_PlusPlus.makeInstance(m_sline, m_schr);
         }
         if( m_text == "--" )
         {
            m_hadOperator = false;
            return parser->T_MinusMinus.makeInstance(m_sline, m_schr);
         }
         if( m_text == "**" ) return parser->T_Power.makeInstance(m_sline, m_schr);
         if( m_text == "==" ) return parser->T_DblEq.makeInstance(m_sline, m_schr);
         if( m_text == "!=" ) return parser->T_NotEq.makeInstance(m_sline, m_schr);
         if( m_text == "<=" ) return parser->T_LE.makeInstance(m_sline, m_schr);
         if( m_text == ">=" ) return parser->T_GE.makeInstance(m_sline, m_schr);
         if( m_text == "=>" ) return parser->T_Arrow.makeInstance(m_sline, m_schr);
         
         if( m_text == "+=" ) return parser->T_AutoAdd.makeInstance(m_sline, m_schr);
         if( m_text == "-=" ) return parser->T_AutoSub.makeInstance(m_sline, m_schr);
         if( m_text == "*=" ) return parser->T_AutoTimes.makeInstance(m_sline, m_schr);
         if( m_text == "/=" ) return parser->T_AutoDiv.makeInstance(m_sline, m_schr);
         if( m_text == "%=" ) return parser->T_AutoMod.makeInstance(m_sline, m_schr);
         if( m_text == ">>" ) return parser->T_RShift.makeInstance(m_sline, m_schr);
         if( m_text == "<<" ) return parser->T_LShift.makeInstance(m_sline, m_schr);
         
         if( m_text == "^&" ) return parser->T_BAND.makeInstance(m_sline, m_schr);
         if( m_text == "^|" ) return parser->T_BOR.makeInstance(m_sline, m_schr);
         if( m_text == "^^" ) return parser->T_BXOR.makeInstance(m_sline, m_schr);
         if( m_text == "^!" ) return parser->T_BNOT.makeInstance(m_sline, m_schr);
         
         if( m_text == "^+" ) return parser->T_OOB.makeInstance(m_sline, m_schr);
         if( m_text == "^-" ) return parser->T_DEOOB.makeInstance(m_sline, m_schr);
         if( m_text == "^%" ) return parser->T_XOOB.makeInstance(m_sline, m_schr);
         if( m_text == "^?" ) return parser->T_ISOOB.makeInstance(m_sline, m_schr);
         
         if( m_text == "^~" ) return parser->T_UNQUOTE.makeInstance(m_sline, m_schr);
         if( m_text == "^." ) return parser->T_COMPOSE.makeInstance(m_sline, m_schr);
         
         if( m_text == "^=" ) return parser->T_EVALRET.makeInstance(m_sline, m_schr);
         if( m_text == "^*" ) return parser->T_EVALRET_EXEC.makeInstance(m_sline, m_schr);
         break;

      case 3:
         if( m_text == "**=" ) return parser->T_AutoPow.makeInstance(m_sline, m_schr);
         if( m_text == "===" ) return parser->T_EEQ.makeInstance(m_sline, m_schr);
         if( m_text == ">>=" ) return parser->T_AutoRShift.makeInstance(m_sline, m_schr);
         if( m_text == "<<=" ) return parser->T_AutoLShift.makeInstance(m_sline, m_schr);
         if( m_text == "^.." ) return parser->T_FUNCPOWER.makeInstance(m_sline, m_schr);
         if( m_text == "*=>" ) return parser->T_STARARROW.makeInstance(m_sline, m_schr);
         break;
   }

   m_hadOperator = false;
   // in case of error
   addError( e_inv_token );

   return 0;
}

void SourceLexer::addNameSpace(const String& ns)
{
   _p->m_nsSet.insert( ns ); 
}

bool SourceLexer::isNameSpace(const String& name)
{
   return _p->m_nsSet.find( name ) != _p->m_nsSet.end();
}

}

/* end of sourcelexer.cpp */
