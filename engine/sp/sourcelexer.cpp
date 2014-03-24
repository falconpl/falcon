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
#include <deque>

namespace Falcon {

class SourceLexer::Private
{
public:
   typedef std::set<String> StringSet;
   typedef std::deque<Parsing::TokenInstance*> NextTokens;
   NextTokens m_nextTokens;
   StringSet m_nsSet;

   typedef enum {
      e_pt_round,
      e_pt_square,
      e_pt_graph
   }
   t_parenthesisType;

   typedef std::deque<t_parenthesisType> ParenthesisStack;
   ParenthesisStack m_pars;
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
   m_outscape(false),
   m_bParsingFtd(false),
   m_state( state_none ),
   m_string_type( e_st_normal ),
   m_bRegexIgnoreCase(true)
{
}

SourceLexer::~SourceLexer()
{
   Private::NextTokens::iterator ti = _p->m_nextTokens.begin();
   Private::NextTokens::iterator tiend = _p->m_nextTokens.end();
   while( ti != tiend ) {
      Parsing::TokenInstance* t = *ti;
      t->dispose();
      ++ti;
   }

   delete _p;
}

Parsing::TokenInstance* SourceLexer::readOutscape()
{
   SourceParser* parser = static_cast<SourceParser*>(m_parser);
   enum {
         e_normal,
         e_esc1,
         e_esc2,
         e_escF,
         e_escA,
         e_escL
   }
   state;

   // clear string
   m_text.clear();

   // in outscape mode, everything up to an escape (or at EOF) is a "fast print" statement.
   char_t chr;
   state = e_normal;

   // if we exited, then we completed a token.
   if( m_chr == 0 )
   {
      // start the count
      m_chr = 1;
      m_line = 1;
   }
   m_sline = m_line;
   m_schr = m_chr;

   while( (chr = m_reader->getChar()) != (char_t)-1 )
   {
      if ( chr == '\n' )
      {
         m_line++;
         m_chr = 1;
      }
      else
      {
         m_chr++;
      }

      switch( state )
      {

         case e_normal:
            if ( chr == '<' )
            {
               state = e_esc1;
            }
            else {
               m_text.append( chr );
            }
         break;

         case e_esc1:
            if ( chr == '?' )
               state = e_esc2;
            else {
               m_text.append( '<' );
               m_text.append( chr );
               state  = e_normal;
            }
         break;

         case e_esc2:
            if ( chr == '=' || String::isWhiteSpace(chr) || chr == '\r' || chr == '\n' )
            {
               // we enter now in the eval mode
               m_outscape = false;
               // generate a >> "text" EOL >>  sequence
               _p->m_nextTokens.push_back( makeString() );
               m_text.clear();
               _p->m_nextTokens.push_back( m_parser->T_EOL.makeInstance(m_line, -1) ); // fake one

               if( chr == '=') {
                  // this will print the evaluation
                  _p->m_nextTokens.push_back( parser->T_RShift.makeInstance(m_line, m_chr) );
               }
               // this will print the text prior the evaluation.
               return parser->T_RShift.makeInstance(m_line, m_chr);
            }
            else if ( chr == 'f' )
            {
               state = e_escF;
            }
            else
            {
               state = e_normal;
               m_text.append( '<' );
               m_text.append( '?' );
               m_text.append( chr );
            }
         break;

         case e_escF:
            if ( chr == 'a' )
            {
               state = e_escA;
            }
            else {
               state = e_normal;
               m_text.append( "<?f" );
               m_text.append( chr );
            }
         break;

         case e_escA:
            if ( chr == 'l' )
            {
              state = e_escL;
            }
            else {
               state = e_normal;
               m_text.append( "<?fa" );
               m_text.append( chr );
            }
         break;

         case e_escL:
            if ( String::isWhiteSpace(chr) || chr == '\r' || chr == '\n' )
            {
               // we enter now the normal mode; we start to consider a standard program.
               m_outscape = false;
               // Return now > and a string, that will be considered a > "..." statement.
               _p->m_nextTokens.push_back( makeString() );
               m_text.clear();
               _p->m_nextTokens.push_back( m_parser->T_EOL.makeInstance(m_line, -1) ); // fake one
               return parser->T_Greater.makeInstance(m_sline, m_schr);
            }
            else {
               state = e_normal;
               m_text.append( "<?fal" );
               m_text.append( chr );
            }
         break;
      }
   }

   // If we're here, the stream is eof.
   // return the last text.
   if( m_text.size() != 0 )
   {
      _p->m_nextTokens.push_back( makeString() );
      m_text.clear();
      _p->m_nextTokens.push_back( m_parser->T_EOL.makeInstance(m_line, -1) ); // fake one
      return parser->T_Greater.makeInstance(m_sline, m_schr);
   }

   // else, generate a last EOL statement.
   m_outscape = false;
   return m_parser->T_EOL.makeInstance(m_line, -1);
}

Parsing::TokenInstance* SourceLexer::makeString()
{
   SourceParser* sp = static_cast<SourceParser*>(m_parser);
   m_stringML = false;

   switch( m_string_type )
   {
   case e_st_normal: return sp->T_String.makeInstance( m_sline, m_schr, m_text );
   case e_st_intl: return sp->T_IString.makeInstance( m_sline, m_schr, m_text );
   case e_st_mutable: return sp->T_MString.makeInstance( m_sline, m_schr, m_text );
   case e_st_regex:
   {
      // read the regex options

      String opts;
      char_t chr = m_reader->getChar();
      while( (chr == 'i' || chr == 'o' || chr == 'l' || chr == 'n') && opts.length() <= 4 )
      {
         opts.append(chr);
         if( chr == 'i' ) {
            // it's useless to add it twice.
            m_bRegexIgnoreCase = false;
         }
         chr = m_reader->getChar();
         m_chr++;
      }

      if( chr != (char_t)-1 )
      {
         m_reader->ungetChar(chr);
      }

      if( m_bRegexIgnoreCase )
      {
         opts.append('i');
      }

      if( opts.length() != 0 )
      {
         m_text.append(0);
         m_text.append(opts);
      }

      return sp->T_RString.makeInstance( m_sline, m_schr, m_text );
   }

   }

   return 0;
}

Parsing::TokenInstance* SourceLexer::nextToken()
{
   if( m_reader == 0)
   {
      return 0;
   }

   if ( !_p->m_nextTokens.empty() )
   {
      Parsing::TokenInstance* inst = _p->m_nextTokens.front();
      _p->m_nextTokens.pop_front();
      return inst;
   }

   if (m_outscape) {
      return readOutscape();
   }

   SourceParser* parser = static_cast<SourceParser*>(m_parser);
   String tempString;
   char_t chr;
   t_state previousState = state_none;
   int curMemChr = 0;

   while( ! m_reader->eof() )
   {
      chr = m_reader->getChar();
      if( chr == (char_t)-1 ) {
         // generate a last fake eol.
         chr = '\n';
         m_line--; // back one line because we'll be accounted.
         m_hadOperator = false;
         m_stringML = false;
      }

      // start the char-line count
      if( m_chr == 0 )
      {
         m_chr = 1;
         m_line = 1;
         if ( chr == '#' )
         {
            m_state = state_shebang1;
            continue;
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
               case '\\': m_hadOperator = true; break;
               default:
                  unget(chr);
                  resetState();
                  break;
            }
            break;

         case state_line:
            // anything will start from here.
            m_sline = m_line;
            m_schr = m_chr;
            m_text.clear();

            switch(chr) {
               case ' ': case '\t': case '\r': /* do nothng */ break;
               case '\n':
               {
                  int32 l = m_line;
                  int32 c = m_chr;
                  // After a real new-line, enter in none-state
                  m_state = state_none;
                  // return only if not an operator at end of line.
                  if( ! eatingEOL() )
                  {
                     m_line++;
                     m_chr = 1;
                     if ( m_hadImport ) {
                        _p->m_nextTokens.push_back( m_parser->T_EOL.makeInstance(l,c) );
                     }
                     m_hadImport = false;
                     return m_parser->T_EOL.makeInstance(l, c);
                  }
                  break;
               }
               
               case ';': {
                  m_hadOperator = true; m_hadImport = false;
                  m_chr++;
                  Parsing::TokenInstance* ti = m_parser->T_EOL.makeInstance(m_line, -1);
                  return ti;
               }

               case ':':
                  chr = m_reader->getChar();
                  if( chr == '?' || chr == ':' )
                  {
                     m_text = ":";
                     m_text.append(chr);
                     return checkOperator();
                  }
                  unget(chr);
                  m_hadOperator = true;
                  return parser->T_Colon.makeInstance(m_line, m_chr++);

               case '\\': m_hadOperator = true; break;
               case ',': m_hadOperator = true; return parser->T_Comma.makeInstance(m_line, m_chr++);
               case '"':  m_string_type = e_st_normal; m_stringML = false; m_stringStart = true; m_state = state_double_string; break;
               case '\'': m_string_type = e_st_normal; m_stringML = false; m_stringStart = true; m_state = state_single_string; break;
               case '0': m_state = state_zero_prefix; break;
               case '(': m_chr++; _p->m_pars.push_back(Private::e_pt_round);
                                   return parser->T_Openpar.makeInstance(m_line,m_chr);
                                   break;
               case ')': m_chr++;
                  resetState();
                  if( _p->m_pars.empty() || _p->m_pars.back() != Private::e_pt_round )
                  {
                     addError(e_par_close_unbal);
                  }
                  else {
                     _p->m_pars.pop_back();
                     return parser->T_Closepar.makeInstance(m_line,m_chr);
                  }
                  break;
               case '[': m_chr++; _p->m_pars.push_back(Private::e_pt_square); return parser->T_OpenSquare.makeInstance(m_line,m_chr); break;
               case ']': m_chr++;
                 resetState();
                 if( _p->m_pars.empty() || _p->m_pars.back() != Private::e_pt_square )
                 {
                    addError(e_square_close_unbal);
                 }
                 else {
                    _p->m_pars.pop_back();
                    return parser->T_CloseSquare.makeInstance(m_line,m_chr);
                 }
                 break;

               case '{': m_chr++; _p->m_pars.push_back(Private::e_pt_graph); return parser->T_OpenGraph.makeInstance(m_line,m_chr); break;
               case '}':
               {
                  m_chr++;
                  if( _p->m_pars.empty() || _p->m_pars.back() != Private::e_pt_graph )
                  {
                     addError(e_graph_close_unbal);
                  }
                  else
                  {
                     _p->m_pars.pop_back();
                     resetState();
                     _p->m_nextTokens.push_back( parser->T_end.makeInstance(m_line,m_chr) );
                     m_hadImport = false;
                     // it's a fake.
                     Parsing::TokenInstance* ti = parser->T_EOL.makeInstance(m_line,-1);
                     return ti;
                  }
               }
               break;

               case '/': previousState = m_state; m_state = state_enterComment; break;

               default:
                  if ( isCipher(chr) )
                  {
                     m_text.append(chr);
                     m_state = state_integer;
                  }
                  else if ( isNameCharacter(chr) )
                  {
                     m_text.append(chr);
                     m_state = state_name;
                  }
                  else
                  {
                     m_text.append( chr );
                     m_state = state_operator;
                  }
                  break;
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
                  m_text.clear(); m_text.append('/');
                  unget(chr);
                  // sline might be ok, but not if we're here from state none.
                  // neutralize next character
                  m_chr--; chr = ' ';
                  m_sline = m_line;
                  m_schr = chr; // where '/' was opened.
                  // the state is operator (probably a division).
                  m_state = state_operator;
                  break;
            }
            /* no break */

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
               case '"': m_chr++; resetState(); return makeString();
               case '\\': m_state = state_double_string_esc; break;

               case '\r':
                  // eat \n
                  if( (chr = m_reader->getChar()) != '\n')
                  {
                     m_reader->ungetChar(chr);
                     continue;
                  }
                  /* no break */

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
                  m_state = state_double_string_nl;
                  break;

               default:
                  m_text.append(chr);
                  break;
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
                  return makeString();

               default:
                  m_text.append(chr);
                  m_state = state_double_string;
                  break;
            }
            break;

         case state_double_string_esc:
            switch( chr )
            {
               case '\\': m_text.append('\\'); m_state = state_double_string; break;
               case '"': m_text.append('"'); m_state = state_double_string; break;
               case '\'': m_text.append('\''); m_state = state_double_string; break;
               case 'n': m_text.append('\n'); m_state = state_double_string; break;
               case 'b': m_text.append('\b'); m_state = state_double_string; break;
               case 't': m_text.append('\t'); m_state = state_double_string; break;
               case 'r': m_text.append('\r'); m_state = state_double_string; break;
               case 'x': case 'X': tempString.size(0); m_state = state_double_string_hex; break;
               case 'B': m_state = state_double_string_bin; break;
               case '0': case 'c': case 'C': tempString.size(0); m_state = state_double_string_octal; break;
               default:
                  m_state = state_double_string;
                  addError(e_inv_esc_sequence);
                  break;
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
                     return makeString();
                  }
               }
               else {
                  // on read failure, it's '<EOF>: will break at next loop
                  resetState();
                  return makeString();
               }
            }
            else
            {
               // eat \r
               if( chr == '\r' && (chr = m_reader->getChar()) != '\n')
               {
                  m_reader->ungetChar(chr);
                  continue;
               }

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

                     if ( ! isNameCharacter(chr) )
                     {
                        unget( chr );
                        return m_parser->T_Int.makeInstance(m_sline, m_schr, 0);
                     }
                     addError(e_inv_num_format);
                  }
                  break;
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
               Parsing::TokenInstance* ti = parser->T_MString.makeInstance( m_sline, m_schr, m_text );
               ti->asString()->toMemBuf();
               return ti;
            }
            else if(chr == '\n' ) {
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
               m_chr = 0;
            }
            else if( chr == '}' ) {
               m_chr++;
               resetState();
               Parsing::TokenInstance* ti = parser->T_MString.makeInstance( m_sline, m_schr, m_text );
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
               int64 retval = 1; // to avoid stupid division by zero in case of errors
               m_text.remove(m_text.length()-1,1);
               if ( ! m_text.parseInt( retval ) )
                  addError( e_inv_num_format );

               _p->m_nextTokens.push_back( parser->T_Dot.makeInstance(m_sline, m_schr) );
               return m_parser->T_Int.makeInstance(m_sline, m_schr, retval);
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
         case state_name_next:
            if (!isNameCharacter(chr))
            {
               // special cases
               if( m_text == "p" && chr == '{' )
               {
                  m_chr++;
                  _p->m_pars.push_back(Private::e_pt_graph);
                  resetState();
                  m_text.clear();
                  return parser->T_OpenProto.makeInstance( m_sline, m_schr );
               }
               else if( m_text == "m" && chr == '{' )
               {
                  m_chr++;
                  _p->m_pars.push_back(Private::e_pt_graph);
                  m_state = state_membuf;
                  m_text.clear();
                  break;
               }
               else if( (m_text == "i" || m_text == "m" || m_text == "r" || m_text == "R" ) && (chr == '"' || chr == '\'') )
               {
                  if( m_text == "i" ) {
                     m_string_type = e_st_intl;
                  }
                  else if( m_text == "m" )
                  {
                     m_string_type = e_st_mutable;
                  }
                  else {
                     m_string_type = e_st_regex;
                     m_bRegexIgnoreCase = m_text == "R";
                  }

                  m_stringML = false;
                  m_stringStart = true;
                  m_state = chr == '"' ? state_double_string : state_single_string;
                  m_text.clear();
                  m_chr++;
                  break;
               }
               // namespace check (NAMESPACE. or x..)
               else if ( chr == '.' )
               {
                  m_state = state_sym_dot;
                  break;
               }
               // A normal symbol
               else
               {
                  unget(chr);
                  resetState();
                  return checkWord();
               }
            }
            m_text.append( chr );
            break;

         case state_sym_dot:
            if( chr != '.' )
            {
               unget(chr);
               // if the current word is not a namespace...
               if( ! isNameSpace( m_text ) )
               {
                  // send the word and add a '.' as next token
                  _p->m_nextTokens.push_back( parser->T_Dot.makeInstance(m_sline, m_schr-1) );

                  resetState();
                  return checkWord();
               }
            }
            // go back to "name" acceptance, and add "." in place of ".."
            m_state = state_name_next;
            m_text.append( '.' );
            break;

         case state_operator:
            if( isTokenLimit( chr ) )
            {
               // special case -- dot/square
               if( m_text == ".")
               {
                  if( chr == '[' )
                  {
                     m_chr++;
                     _p->m_pars.push_back(Private::e_pt_square);
                     resetState();
                     return parser->T_DotSquare.makeInstance(m_sline, m_schr );
                  }
                  else if ( chr == '(' )
                  {
                     m_chr++;
                     _p->m_pars.push_back(Private::e_pt_round);
                     resetState();
                     return parser->T_DotPar.makeInstance(m_sline, m_schr );
                  }
               }
               else if( m_text == "^" ) {
                  if ( chr == '(' )
                  {
                     m_chr++;
                     _p->m_pars.push_back(Private::e_pt_round);
                     resetState();

                     return parser->T_CapPar.makeInstance(m_sline, m_schr );
                  }
                  else if( chr == '[' )
                  {
                     m_chr++;
                     _p->m_pars.push_back(Private::e_pt_square);
                     resetState();

                     return parser->T_CapSquare.makeInstance(m_sline, m_schr );
                  }
                  else if( chr == '^' )
                  {
                     m_chr++;
                     resetState();
                     return parser->T_BXOR.makeInstance(m_sline, m_schr );
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
            else if( m_bParsingFtd && m_text == "?" && chr == '>') {
               m_chr++;
               resetState();
               m_outscape = true;
               m_text.clear();
               // generate a fake EOL to exit cleanly from parsing
               return parser->T_EOL.makeInstance(m_sline, -1 );
            }
            else if( m_text != "^" && chr == '.' )
            {
               unget(chr);
               resetState();
               Parsing::TokenInstance* ti = checkOperator();
               if( ti != 0 )
               {
                  return ti;
               }
            }

            m_text.append( chr );
            break;
      }

      //--------------------------------------------------------------
      // Character +  line advance
      // Do this now so we can advance only after having read a char.
      if( chr == '\n' )
      {
         TRACE2( "SourceLexer::nextToken - \\n at line %d -- %s", m_line, ( eatingEOL() ? "Eating away" : "returning") );
         m_line++;
         // are we in a state that requires to go on?
         if( ! eatingEOL() )
         {
            m_chr = 1;
            return parser->T_EOL.makeInstance(m_sline, m_schr);
         }
         // else go on
         m_chr =0;
      }

      m_chr++;
   }

   //=================================
   // End of file control
   //=================================

   if( ! _p->m_pars.empty() )
   {
      switch( _p->m_pars.back() )
      {
      case Private::e_pt_round: addError(e_par_unbal); break;
      case Private::e_pt_square: addError(e_square_unbal); break;
      case Private::e_pt_graph: addError(e_graph_unbal); break;
      }
   }

   if ( m_state == state_line )
   {
      m_state = state_none;
      m_hadImport = false;
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


bool SourceLexer::eatingEOL()
{
   return m_stringML || (m_hadOperator && ! m_hadImport)
            || (!_p->m_pars.empty() && _p->m_pars.back() == Private::e_pt_round);
}

void SourceLexer::resetState()
{
   m_state = state_line;
   m_hadOperator = false; // checkOperator will set it back as needed.
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
         if ( m_text == "loop" ) return parser->T_loop.makeInstance(m_sline, m_schr);
         
         /*
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
         if ( m_text == "switch" ) return parser->T_switch.makeInstance(m_sline, m_schr);
         if ( m_text == "select" ) return parser->T_select.makeInstance(m_sline, m_schr);
         if ( m_text == "return" ) return parser->T_return.makeInstance(m_sline, m_schr);
         if ( m_text == "export" ) return parser->T_export.makeInstance(m_sline, m_schr);
         if ( m_text == "global" ) return parser->T_global.makeInstance(m_sline, m_schr);
         if ( m_text == "object" ) return parser->T_object.makeInstance(m_sline, m_schr);
         if ( m_text == "static" ) return parser->T_static.makeInstance(m_sline, m_schr);

         if ( m_text == "import" ) 
         {
            m_hadImport = true;
            return parser->T_import.makeInstance(m_sline, m_schr);
         }

      break;

      case 7:
         if ( m_text == "forlast" ) return parser->T_forlast.makeInstance(m_sline, m_schr);
         if ( m_text == "finally" ) return parser->T_finally.makeInstance(m_sline, m_schr);
         if ( m_text == "default" ) return parser->T_default.makeInstance(m_sline, m_schr);
      break;


      case 8:
         if ( m_text == "provides" ) return parser->T_provides.makeInstance(m_sline, m_schr);
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
   if( !m_hadImport && isNameSpace(m_text) && m_state == state_name )
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
         if( m_text == ">" ) {m_hadOperator = false; return parser->T_Greater.makeInstance(m_sline, m_schr);}
         if( m_text == "." ) return parser->T_Dot.makeInstance(m_sline, m_schr);
         if( m_text == ":" ) return parser->T_Colon.makeInstance(m_sline, m_schr);
         if( m_text == "," ) return parser->T_Comma.makeInstance(m_sline, m_schr);
         if( m_text == "$" ) return parser->T_Dollar.makeInstance(m_sline, m_schr);
         if( m_text == "&" ) return parser->T_Amper.makeInstance(m_sline, m_schr);
         if( m_text == "#" ) return parser->T_NumberSign.makeInstance(m_sline, m_schr);
         if( m_text == "@" ) return parser->T_At.makeInstance(m_sline, m_schr);
         if( m_text == "?" ) return parser->T_QMark.makeInstance(m_sline, m_schr);
         if( m_text == "~" ) return parser->T_Tilde.makeInstance(m_sline, m_schr);
         if( m_text == "|" ) return parser->T_Disjunct.makeInstance(m_sline, m_schr);
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
         if( m_text == ">>" ) { m_hadOperator = false; return parser->T_RShift.makeInstance(m_sline, m_schr);}
         if( m_text == "<<" ) return parser->T_LShift.makeInstance(m_sline, m_schr);
         
         if( m_text == "^&" ) return parser->T_BAND.makeInstance(m_sline, m_schr);
         if( m_text == "^|" ) return parser->T_BOR.makeInstance(m_sline, m_schr);
         if( m_text == "^^" ) return parser->T_BXOR.makeInstance(m_sline, m_schr);
         if( m_text == "^!" ) return parser->T_BNOT.makeInstance(m_sline, m_schr);
         
         if( m_text == "^+" ) return parser->T_OOB.makeInstance(m_sline, m_schr);
         if( m_text == "^-" ) return parser->T_DEOOB.makeInstance(m_sline, m_schr);
         if( m_text == "^%" ) return parser->T_XOOB.makeInstance(m_sline, m_schr);
         if( m_text == "^$" ) return parser->T_ISOOB.makeInstance(m_sline, m_schr);
         
         if( m_text == "^~" ) return parser->T_UNQUOTE.makeInstance(m_sline, m_schr);
         if( m_text == "^." ) return parser->T_COMPOSE.makeInstance(m_sline, m_schr);
         
         if( m_text == "^=" ) return parser->T_EVALRET.makeInstance(m_sline, m_schr);
         if( m_text == "^?" ) return parser->T_EVALRET_DOUBT.makeInstance(m_sline, m_schr);

         if( m_text == "::" ) return parser->T_DoubleColon.makeInstance(m_sline, m_schr);
         if( m_text == ":?" ) return parser->T_ColonQMark.makeInstance(m_sline, m_schr);

         // outscaping?
         if( m_bParsingFtd && m_text == "?>" ) {
            m_outscape = true;
            return parser->T_EOL.makeInstance(m_line, -1);
         }
         break;

      case 3:
         if( m_text == "^=&" ) return parser->T_EVALRET_EXEC.makeInstance(m_sline, m_schr);
         if( m_text == "**=" ) return parser->T_AutoPow.makeInstance(m_sline, m_schr);
         if( m_text == "===" ) return parser->T_EEQ.makeInstance(m_sline, m_schr);
         if( m_text == ">>=" ) return parser->T_AutoRShift.makeInstance(m_sline, m_schr);
         if( m_text == "<<=" ) return parser->T_AutoLShift.makeInstance(m_sline, m_schr);
         if( m_text == "&=>" ) return parser->T_ETAARROW.makeInstance(m_sline, m_schr);
         break;
   }

   m_hadOperator = false;
   // in case of error
   addError( e_inv_token );

   return 0;
}

void SourceLexer::addNameSpace(const String& ns)
{
   // insert also all the namespace components.
   uint32 pos = ns.find('.');
   while( pos != String::npos )
   {
      _p->m_nsSet.insert( ns.subString(0,pos) );
      pos = ns.find('.', pos+1);
   }
   _p->m_nsSet.insert( ns ); 
}

bool SourceLexer::isNameSpace(const String& name)
{
   return _p->m_nsSet.find( name ) != _p->m_nsSet.end();
}

void SourceLexer::setParsingFam( bool m )
{
   if( m ) {
      m_outscape = true;
      m_bParsingFtd = true;
   }
   else {
      m_bParsingFtd = false;
   }
}

}

/* end of sourcelexer.cpp */
