/*
   FALCON - The Falcon Programming Language.
   FILE: streamtok.cpp

   Advanced general purpose stream tokenizer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 23 Apr 2014 10:45:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/streamtok.cpp"

#include <falcon/streamtok.h>
#include <falcon/fassert.h>
#include <falcon/textreader.h>
#include <falcon/mt.h>
#include <falcon/stream.h>

#include "../re2/re2/re2.h"
#include "../re2/re2/stringpiece.h"

#include <vector>

namespace Falcon {

namespace {

class Token
{
public:
   int32 m_id;
   re2::RE2* m_regex;
   re2::StringPiece m_result;

   // we'll work with utf8 only
   char* m_token;
   length_t m_tokenLen;
   void* m_data;

   Token( int32 id, re2::RE2* regex, void* data = 0, StreamTokenizer::deletor del = 0 ):
      m_id(id),
      m_regex(regex),
      m_token(0),
      m_data(data),
      m_deletor(del)
   {}

   Token( int32 id, const String& token, void* data = 0, StreamTokenizer::deletor del = 0):
      m_id(id),
      m_regex(0),
      m_token(0),
      m_data(data),
      m_deletor(del)
   {
      // convert the token in utf8
      m_token = token.toUTF8String(m_tokenLen);
   }

   ~Token();

private:
   StreamTokenizer::deletor m_deletor;
};


Token::~Token()
{
   if ( m_data != 0 && m_deletor != 0 )
   {
      m_deletor(m_data);
   }

   delete m_regex;
   delete[] m_token;
}

}


class StreamTokenizer::Private
{
public:
   typedef std::vector<Token*> TokenList;
   TokenList m_tlist;

   Private():
      m_refCount(1),
      m_resData(0),
      m_resDataDel(0),
      m_resTokData(0),
      m_resTokDataDel(0)
   {}

   ~Private() {
      clear();

      if ( m_resDataDel != 0 )
      {
         m_resDataDel(m_resData);
      }

      if ( m_resTokDataDel != 0 )
      {
         m_resTokDataDel(m_resTokData);
      }
   }

   void clear()
   {
      TokenList::iterator iter = m_tlist.begin();
      while( iter != m_tlist.end() )
      {
         delete *iter;
         ++iter;
      }

      m_tlist.clear();
   }

   Mutex m_mtx;
   int m_refCount;
   void* m_resData;
   StreamTokenizer::deletor m_resDataDel;

   void* m_resTokData;
   StreamTokenizer::deletor m_resTokDataDel;
};



StreamTokenizer::StreamTokenizer( TextReader* source, uint32 bufsize )
{
   m_bufSize = bufsize;
   init();

   setSource( source );
}


StreamTokenizer::StreamTokenizer( const String& buffer, uint32 bufsize )
{
   m_bufSize = bufsize;
   init();

   setSource(buffer);
}


StreamTokenizer::StreamTokenizer( const StreamTokenizer& other )
{
   other._p->m_mtx.lock();
   m_tr = other.m_tr;
   if( m_tr !=0 )
   {
      m_tr->incref();
   }

   m_bufSize = other.m_bufSize;
   m_bufLen = other.m_bufLen;
   m_buffer = new char[other.m_bufSize];
   if ( m_bufLen != 0 )
   {
      memcpy( m_buffer, other.m_buffer, m_bufLen );
   }

   m_srcPos = other.m_srcPos;
   m_bOwnText = other.m_bOwnText;
   m_bOwnToken = other.m_bOwnToken;
   m_hasRegex = other.m_hasRegex;

   m_giveTokens = other.m_giveTokens;
   m_groupTokens = other.m_groupTokens;
   m_maxTokenSize = 0;

   _p = other._p;
   _p->m_refCount++;
   _p->m_mtx.unlock();
}


StreamTokenizer::~StreamTokenizer()
{
   if ( m_tr != 0 )
   {
      m_tr->decref();
   }

   delete[] m_buffer;

   _p->m_mtx.lock();
   if( --_p->m_refCount == 0 )
   {
      _p->m_mtx.unlock();
      delete _p;
   }
   else {
      _p->m_mtx.unlock();
   }
}


void StreamTokenizer::init()
{
   m_bOwnText = true;
   m_bOwnToken = true;
   m_hasRegex = false;
   m_giveTokens = false;
   m_groupTokens = false;
   m_buffer = new char[m_bufSize];
   m_maxTokenSize = 0;

   m_tr = 0;
   m_tokenID = 0;
   _p = new Private;
}


void StreamTokenizer::reinit()
{
   m_hasLastToken = true;
   m_tokenLoaded = false;
   m_hasToken = false;
   m_srcPos = 0;
   m_bufLen = 0;
}


void StreamTokenizer::setSource(TextReader* source )
{
   source->incref();

   _p->m_mtx.lock();
   if( m_tr != 0 )
   {
      TextReader* old = m_tr;
      m_tr = 0;
      _p->m_mtx.unlock();
      old->decref();
      _p->m_mtx.lock();
   }

   m_tr = source;
   reinit();
   _p->m_mtx.unlock();
}


void StreamTokenizer::setSource(const String& buffer )
{
   _p->m_mtx.lock();
   if( m_tr != 0 )
   {
      TextReader* old = m_tr;
      m_tr = 0;
      _p->m_mtx.unlock();
      old->decref();
      _p->m_mtx.lock();
   }

   m_source = buffer;
   m_source.bufferize();
   reinit();
   _p->m_mtx.unlock();
}


void StreamTokenizer::addToken( const String& token, void* data, StreamTokenizer::deletor del )
{
   Token* tk = new Token(_p->m_tlist.size(), token, data, del );
   _p->m_mtx.lock();
   if( m_maxTokenSize < tk->m_tokenLen  )
   {
      m_maxTokenSize = tk->m_tokenLen;
   }

   _p->m_tlist.push_back(tk);
   _p->m_mtx.unlock();
}


void StreamTokenizer::addRE( const String& re, void* data, StreamTokenizer::deletor del )
{
   // this might throw
   re2::RE2* regex = new re2::RE2(re);

   Token* tk = new Token(_p->m_tlist.size(), regex, data, del );
   _p->m_mtx.lock();
   _p->m_tlist.push_back(tk);
   _p->m_mtx.unlock();
}


void StreamTokenizer::addRE( re2::RE2* regex, void* data, StreamTokenizer::deletor del )
{
   // this might throw
   Token* tk = new Token(_p->m_tlist.size(), regex, data, del );
   _p->m_mtx.lock();
   _p->m_tlist.push_back(tk);
   _p->m_mtx.unlock();
}



void StreamTokenizer::setTextCallbackData(void* data, StreamTokenizer::deletor del)
{
   _p->m_mtx.lock();
   _p->m_resData = data;
   _p->m_resDataDel = del;
   _p->m_mtx.unlock();
}


void StreamTokenizer::setTokenCallbackData(void* data, StreamTokenizer::deletor del)
{
   _p->m_mtx.lock();
   _p->m_resTokData = data;
   _p->m_resTokDataDel = del;
   _p->m_mtx.unlock();
}


void* StreamTokenizer::getTextCallbackData() const
{
   _p->m_mtx.lock();
   void* data = _p->m_resData;
   _p->m_mtx.unlock();

   return data;
}


void* StreamTokenizer::getTokenCallbackData() const
{
   _p->m_mtx.lock();
   void* data = _p->m_resTokData;
   _p->m_mtx.unlock();

   return data;
}


void StreamTokenizer::clearTokens()
{
   _p->m_mtx.lock();
   _p->clear();
   _p->m_mtx.unlock();
}



void StreamTokenizer::rewind()
{
   _p->m_mtx.lock();
   if( m_tr != 0 )
   {
      m_bufLen = 0;
      Stream* stream = m_tr->underlying();
      Transcoder* tc = m_tr->transcoder();
      // this might throw
      stream->seekBegin(0);

      stream->incref();
      m_tr->decref();

      m_tr = new TextReader(stream, tc);
      stream->decref();

   }
   reinit();
   _p->m_mtx.unlock();
}


bool StreamTokenizer::getNextChar( char_t &chr )
{
   if( m_tr != 0 )
   {
      if( m_tr->eof() ) return false;
      chr = m_tr->getChar();
      return ! m_tr->eof();
   }
   else if( m_srcPos < m_source.length() )
   {
      chr = m_source[m_srcPos++];
      return true;
   }
   return false;
}


bool StreamTokenizer::next()
{
   String currentToken;

   _p->m_mtx.lock();
   // if we recorded a token, then we're bound to give it back.
   if( m_tokenLoaded )
   {
      Token* tk = _p->m_tlist[m_tokenID];
      m_tokenLoaded = false;
      String* token = new String(m_lastToken);
      void* data = tk->m_data == 0 ? _p->m_resTokData : tk->m_data;
      _p->m_mtx.unlock();

      onTokenFound(m_tokenID, token, data);

      if( m_bOwnToken )
      {
         delete token;
      }
      // there is *ALWAYS* something after a token.
      return true;
   }

   char_t nextChar;
   String* running = new String;
   if ( ! m_hasLastToken )
   {
      _p->m_mtx.unlock();
      return false;
   }

   m_hasLastToken = false;
   bool readNext = m_bufLen == 0;

   while( true )
   {
      if( readNext )
      {
         if( ! getNextChar(nextChar) )
         {
            // the running text is m_bufPos -> buflen (might be 0)
            running->fromUTF8(m_buffer, m_bufLen );
            // do not declare having found a token.
            break;
         }

         if( m_bufLen + 5 >= m_bufSize )
         {
            m_bufSize *= 2;
            char* newMem = new char[m_bufSize];
            memcpy(newMem, m_buffer, m_bufLen);
            delete m_buffer;
            m_buffer = newMem;
         }

         length_t utf8Len = String::charToUTF8(nextChar, m_buffer + m_bufLen );
         m_bufLen += utf8Len;
      }
      readNext = true;

      int32 id = 0;
      length_t pos = 0;
      if( findNext(id, pos) )
      {
         Token* tk = _p->m_tlist[id];
         uint32 toklen = tk->m_result.length();

         if( m_giveTokens || m_groupTokens )
         {
            currentToken.fromUTF8(m_buffer + pos, toklen );

            if( m_groupTokens && pos == 0 && m_hasToken ) {
               // retry?
               if( currentToken == m_lastToken )
               {
                  if( m_bufLen > toklen )
                  {
                     readNext = false;
                     memcpy( m_buffer, m_buffer + toklen, m_bufLen - toklen );
                  }
                  m_bufLen -= toklen;
                  continue;
               }
            }
            m_lastToken = currentToken;

            if( m_giveTokens )
            {
               m_tokenLoaded = true;
               m_tokenID = id;
            }
         }

         // the running text is m_bufPos -> pos
         running->fromUTF8(m_buffer, pos);

         // inform hasNext that we have at least an empty string to return.
         m_hasLastToken = true;
         m_hasToken = true;

         uint32 tokend = pos + toklen;
         if( m_bufLen > tokend )
         {
            memcpy( m_buffer, m_buffer + tokend, m_bufLen - tokend );
         }
         m_bufLen -= tokend;
         break;
      }
   }

   void* resData = _p->m_resData;
   bool bOwnText = m_bOwnText;
   _p->m_mtx.unlock();

   onTextFound( running, resData );

   if( bOwnText )
   {
      delete running;
   }

   return true;
}


bool StreamTokenizer::hasNext() const
{
   bool res;

   // if we have a text reader, the thing is over when we hit eof.
   _p->m_mtx.lock();

   if( m_tr != 0 )
   {
      // and of course, when we also have exhausted our buffer.
      res = ! m_tr->eof();
   }
   else {
      res = m_srcPos < m_source.length();
   }
   res |= m_tokenLoaded || m_hasLastToken;
   _p->m_mtx.unlock();

   return res;
}


uint32 StreamTokenizer::currentMark() const
{
   return m_mark;
}


void StreamTokenizer::gcMark( uint32 mark )
{
   m_mark = mark;
}


bool StreamTokenizer::findNext( int32& id, length_t& pos )
{
   Private::TokenList::const_iterator iter = _p->m_tlist.begin();
   Private::TokenList::const_iterator end = _p->m_tlist.end();

   id = -1;
   pos = String::npos;

   // this is a very fast op when m_buffer is char*
   re2::StringPiece spc(m_buffer, (int)m_bufLen);

   while( iter != end )
   {
      Token* tk = *iter;
      length_t foundAt = String::npos;

      if( tk->m_token != 0 )
      {
         foundAt = findText( tk->m_token, tk->m_tokenLen );
         // we have a match, but can we trust it?
         if( foundAt < pos )
         {
            if (m_tr == 0 || m_tr->eof() || m_bufLen-foundAt >=  m_maxTokenSize)
            {
               tk->m_result.set(tk->m_token, (int)tk->m_tokenLen);
               pos = foundAt;
               id = tk->m_id;
            }
         }
      }
      else {
         fassert( tk->m_regex != 0 );
         if( tk->m_regex->Match( spc, 0, m_bufLen, re2::RE2::UNANCHORED, &tk->m_result, 1 ) )
         {
            foundAt = static_cast<length_t>(tk->m_result.begin() - spc.begin());
            // we have a match, but can we trust it?
            if( foundAt < pos )
            {
               // if we have a string buffer, we're at EOF or we read 1 char past the lenght of the token, we found it.
               if ( m_tr == 0 || m_tr->eof() || spc.end() > tk->m_result.end() )
               {
                  pos = foundAt;
                  id = tk->m_id;
               }
            }
         }
      }


      ++iter;
   }

   return id != -1;
}


length_t StreamTokenizer::findText( const char* token, length_t len )
{
   if( len > m_bufLen )
   {
      return String::npos;
   }

   length_t tpos;
   if( m_bufLen >= m_maxTokenSize )
   {
      tpos = m_bufLen - m_maxTokenSize;
   }
   else {
      tpos = 0;
   }
   char* buf = m_buffer + tpos;

   while( len > 0 )
   {
      if ( *buf != *token )
      {
         return String::npos;
      }

      ++buf;
      ++token;
      --len;
   }

   return tpos;
}


bool StreamTokenizer::giveTokens() const
{
   _p->m_mtx.lock();
   bool b = m_giveTokens;
   _p->m_mtx.unlock();
   return b;
}


bool StreamTokenizer::groupTokens() const
{
   _p->m_mtx.lock();
   bool b = m_groupTokens;
   _p->m_mtx.unlock();
   return b;
}


void StreamTokenizer::giveTokens(bool b)
{
   _p->m_mtx.lock();
   m_giveTokens = b;
   _p->m_mtx.unlock();
}


void StreamTokenizer::groupTokens(bool b)
{
   _p->m_mtx.lock();
   m_groupTokens = b;
   _p->m_mtx.unlock();
}


void StreamTokenizer::setOwnText( bool mode )
{
   _p->m_mtx.lock();
   m_bOwnText= mode;
   _p->m_mtx.unlock();
}


bool StreamTokenizer::ownText() const
{
   _p->m_mtx.lock();
   bool b = m_bOwnText;
   _p->m_mtx.unlock();
   return b;
}


void StreamTokenizer::setOwnToken( bool mode )
{
   _p->m_mtx.lock();
   m_bOwnToken = mode;
   _p->m_mtx.unlock();
}


bool StreamTokenizer::ownToken() const
{
   _p->m_mtx.lock();
   bool b = m_bOwnToken;
   _p->m_mtx.unlock();
   return b;
}

}

/* end of streamtok.cpp */
