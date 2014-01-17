/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.cpp

   Text-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/textreader.h>
#include <falcon/stream.h>
#include <falcon/engine.h>
#include <falcon/transcoder.h>
#include <falcon/stderrors.h>

#include <string.h>

#include <list>


namespace Falcon {

TextReader::TextReader( Stream* stream ):
   Reader( stream ),
   m_pushedChr((char_t)-1),
   m_cTokens( 0 ),
   m_cTSize( 0 ),
   m_mark(0)
{
   m_decoder = Engine::instance()->getTranscoder("C");
   makeDefaultSeps();
}


TextReader::TextReader( Stream* stream, Transcoder* decoder ):
   Reader( stream ),
   m_pushedChr(-1),
   m_decoder( decoder ),
   m_cTokens( 0 ),
   m_cTSize( 0 ),
   m_mark(0)
{
   makeDefaultSeps();
}


TextReader::~TextReader()
{
   clearTokens();
}

void TextReader::clearTokens()
{
   for( int count = 0; count < m_cTSize; ++count )
   {
      delete[] m_cTokens[count].data;
   }

   delete[] m_cTokens;
   m_cTSize = 0;
   m_cTokens = 0;
}


void TextReader::setEncoding( Transcoder* decoder )
{
   m_decoder = decoder;
   makeDefaultSeps();
}


void TextReader::makeDefaultSeps()
{
   m_nl[0].size= m_decoder->encode( "\n", m_nl[0].data, sizeof(m_nl[0].data));
   m_nl[1].size= m_decoder->encode( "\r\n", m_nl[1].data, sizeof(m_nl[1].data));
   m_nl[2].size= m_decoder->encode( "\n\r", m_nl[2].data, sizeof(m_nl[2].data));
}


bool TextReader::read( String& str, length_t count, length_t start )
{
   fassert( m_stream != 0 );
   fassert( m_decoder != 0 );
   
   // bootstrap
   if( count <= 0 || m_stream->eof() )
   {
      return true;
   }

   str.reserve(count + start);
   str.size(start * str.manipulator()->charSize());

   length_t done = 0;
   if( m_pushedChr != (char_t)-1)
   {
      str.append( m_pushedChr );
      m_pushedChr = (char_t)-1;
      done++;
   }


   length_t more = count - done;
   while( done < count )
   {
      String temp;

      // read in more data --
      // -- the data we should read MIGHT be more than count, but for sure, isn't less
      if( ! fetch( more ) )
      {
         // Fetch returns false when there is nothing more to be processsed in the stream.
         break;
      }

      length_t decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos, temp, count-done, true );
      
      // was the decoder able to get something?
      if( decoded == 0 )
      {
         // We must be close to our goal, but we need some more characters.
         // -- 8 is a good guess for a last loop under any encoding.
         more = 8;
      }
      else
      {
         m_bufPos += decoded;
         done += temp.length();
         str += temp;
         // read a smaller amount of data.
         // -- actually, the data read in is usually much wider than this.
         more = count - done;
      }
   }

   // return true if we did process anything
   return done > 0;
}


bool TextReader::read( String& str )
{
   return read( str, str.allocated(), 0 );
}


bool TextReader::readEof( String& str )
{
   std::list<String> strList;
   length_t totalSize = 0;
   
   while( fetch(1024) )
   {
      strList.push_back("");
      String& temp = strList.back();
      length_t decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos, temp, 1024, true );
      m_bufPos += decoded;
      totalSize += temp.length();
   }

   if( totalSize == 0 )
   {
      return false;
   }

   str.reserve( str.length() + totalSize );
   std::list<String>::iterator iter = strList.begin();
   while( iter != strList.end() )
   {
      str += *iter;
      ++iter;
   }

   return true;
}


bool TextReader::readLine( String& target, length_t maxCount )
{
   // prepare the new line tokens.
   struct token toks[3];
   toks[0].data = m_nl[0].data;
   toks[0].size = m_nl[0].size;
   toks[1].data = m_nl[1].data;
   toks[1].size = m_nl[1].size;
   toks[2].data = m_nl[2].data;
   toks[2].size = m_nl[2].size;
   
   return readTokenInternal( target, toks, 3, maxCount ) >= 0;
}


bool TextReader::readRecord( String& target, const String& separator, length_t maxCount )
{
   struct token tok;
   byte alloc[128];
   length_t expSize = m_decoder->encodingSize(separator.length()+4);
   if ( expSize > 128 )
   {
      tok.data = new byte[expSize];
   }
   else
   {
      tok.data = alloc;
   }


   try
   {
      tok.size = m_decoder->encode(separator, tok.data, expSize );
      int t = readTokenInternal( target, &tok, 1, maxCount );
      if( tok.data != alloc )
      {
         delete[] tok.data;
      }
      return t == 0;
   }
   catch(...)
   {
      if( tok.data != alloc )
      {
         delete[] tok.data;
      }
      throw;
   }
}


int TextReader::readToken( String& target, const String* tokens, int32 tokenCount, length_t maxCount )
{
   if( tokens != 0 )
   {
      clearTokens();
      m_cTokens = new struct token[tokenCount];
      for( int i = 0; i < tokenCount; ++i )
      {
         length_t size = tokens[i].length();
         m_cTokens[i].data = new byte[m_decoder->encodingSize(size)];
         m_cTSize++;

         // shouldn't throw, but if it does, clearToken() will take care of clearing
         // up to m_cTSize.
         m_cTokens[i].size = m_decoder->encode( tokens[i], m_cTokens[i].data, size );
      }
   }

   if ( m_cTokens == 0 )
   {
      throw new EncodingError(ErrorParam(e_dec_fail, __LINE__, __FILE__).extra("Missing tokens"));
   }

   return readTokenInternal( target, m_cTokens, m_cTSize, maxCount );
}


int TextReader::readTokenInternal( String& target, struct token* tokens, int32 tokenCount, length_t maxCount )
{
   if ( m_bufPos == m_bufLength && m_stream->eof() )
   {
      return -1;
   }
   
   target.size(0);
   target.reserve(maxCount);

   // try to get in as much data as possible
   while( target.length() < maxCount )
   {
      String temp;
      length_t decoded;
      
      if( ! fetch( m_decoder->encodingSize(maxCount - target.length())) )
      {
         // eof? -- return the last token.
         return target.size() == 0 ? -1 : 1;
      }

      length_t pos = m_bufPos;
      int tok = findFirstToken( tokens, tokenCount, pos );

      if( tok >= 0 )
      {
         // found!
         if( pos == m_bufPos )
         {
            // discard the token
            m_bufPos += tokens[tok].size;
             // we're done here -- what was in target is OK for us.
            return tok;
         }
         else
         {
            // use target if possible.
            if( target.size() == 0 )
            {               
               decoded = m_decoder->decode(m_buffer + m_bufPos, pos-m_bufPos,
                                                 target, maxCount - target.length(), true );
            }
            else
            {
               // else accumulate the string.
               decoded = m_decoder->decode(m_buffer + m_bufPos, pos-m_bufPos,
                                                 temp, maxCount - target.length(), true );
               target += temp;
            }

            // update the position
            m_bufPos += decoded;

            // if we decoded exactly up to the position of the separator,
            // that must be discarded
            if( pos == m_bufPos)
            {
               m_bufPos += tokens[tok].size;
               // we're done here.
               return tok;
            }
            // else, we did find the token, but the target space was over before
            // than we could read everything.
            return -1;
         }

        
      }
      // token not found. Decode the whole buffer, and eventually try again.
      else
      {
        // use target if possible.
         if( target.size() == 0 )
         {
            decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos,
                                              target, maxCount, true );
         }
         else
         {
            // else accumulate the string.
            decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos,
                                              temp, maxCount - target.length(), true );
            target += temp;
         }
         
         m_bufPos += decoded;
      }
   }

   // if we're here, we read something, but it wasn't a complete line.
   return -1;
}


int TextReader::findFirstToken( struct token* toks, int tcount, length_t& pos )
{
   // find the position in the buffer where the token is to be found.
   byte* base = m_buffer + m_bufPos;
   byte* buf = m_buffer + m_bufPos;
   byte* maxbuf = m_buffer + m_bufLength;

   while( buf < maxbuf )
   {
      for( int i = 0; i < tcount; ++i )
      {
         struct token& tok = toks[i];
         if( buf + tok.size <= maxbuf && tok.data[0] == *buf )
         {
            if ( tok.size == 1 || memcmp(tok.data, buf, tok.size ) == 0 )
            {
               pos = (length_t)(buf - base)+m_bufPos;
               return i;
            }
         }
      }

      ++buf;
   }

   // not found
   return -1;
}


char_t TextReader::getChar()
{
   if( m_pushedChr != NoChar )
   {
      char_t tmp = m_pushedChr;
      m_pushedChr = NoChar;
      return tmp;
   }

   String str;
   length_t decoded = 0;
   // do we have enough space?
   if( m_bufPos < m_bufLength )
   {
      decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos, str, 1, true );
   }

   // not enough length or decode failed?
   if( decoded == 0 )
   {
      // try again refetching -- if the failure may be a close call.
      if( m_bufPos + 4 >= m_bufLength )
      {
         if( fetch(4) )
         {
            decoded = m_decoder->decode(m_buffer + m_bufPos, m_bufLength-m_bufPos, str, 1, true );
         }
      }

      if( decoded == 0 )
      {
         return (char_t)-1;
      }
   }

   m_bufPos += decoded;
   return str.getCharAt(0);
}


void TextReader::ungetChar( char_t chr )
{
   m_pushedChr = chr;
}

}

/* end of textreader.cpp */
