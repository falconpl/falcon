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
      m_resDataDel(0)
   {}

   ~Private() {
      TokenList::iterator iter = m_tlist.begin();
      while( iter != m_tlist.end() )
      {
         delete *iter;
         ++iter;
      }

      if ( m_resDataDel != 0 )
      {
         m_resDataDel(m_resData);
      }
   }

   Mutex m_mtx;
   int m_refCount;
   void* m_resData;
   StreamTokenizer::deletor m_resDataDel;
};



StreamTokenizer::StreamTokenizer( TextReader* source, uint32 bufsize )
{
   m_tr = source;
   m_bufSize = bufsize*4+4;
   m_bufLen = 0;
   m_buffer = 0;
   init();
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

   m_bufPos = other.m_bufPos;
   m_bOwnText = other.m_bOwnText;
   m_bOwnToken = other.m_bOwnToken;
   m_hasRegex = other.m_hasRegex;

   _p = other._p;
   _p->m_refCount++;
   _p->m_mtx.unlock();
}

StreamTokenizer::StreamTokenizer( const String& buffer )
{
   m_tr = 0;
   m_buffer = buffer.toUTF8String(m_bufSize);
   m_bufLen = m_bufSize;
   init();
}


StreamTokenizer::~StreamTokenizer()
{
   if ( m_tr != 0 )
   {
      m_tr->decref();
   }

   delete m_buffer;

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
   m_bufPos = 0;
   m_bOwnText = true;
   m_bOwnToken = true;
   m_hasRegex = false;

   _p = new Private;
}


void StreamTokenizer::addToken( const String& token, void* data, StreamTokenizer::deletor del )
{
   Token* tk = new Token(_p->m_tlist.size(), token, data, del );
   _p->m_mtx.lock();
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


void StreamTokenizer::onResidual(void* data, StreamTokenizer::deletor del)
{
   _p->m_mtx.lock();
   _p->m_resData = data;
   _p->m_resDataDel = del;
   _p->m_mtx.unlock();
}


void StreamTokenizer::onTokenFound( int32 , String* , String* , void* )
{
   // Do Nothing
}

bool StreamTokenizer::next()
{
   _p->m_mtx.lock();
   if( m_bufPos == m_bufLen )
   {
      if ( ! refill() )
      {
         _p->m_mtx.unlock();
         return false;
      }
   }

   int32 id = 0;
   length_t pos = 0;
   if( findNext(id, pos) )
   {
      // the running text is m_bufPos -> pos
      String* running = new String;
      running->fromUTF8(m_buffer + m_bufPos, pos - m_bufPos);

      // the token content can be either static or a result of the regex
      Token* tk = _p->m_tlist[id];

      pos += (tk->m_token != 0 ?
               tk->m_tokenLen :
               static_cast<length_t>(tk->m_result.end() - tk->m_result.begin()) );
      if( m_bufPos == pos )
      {
         m_bufPos++;
      }
      else
      {
         m_bufPos = pos;
      }
      _p->m_mtx.unlock();

      String *tok = new String;
      if( tk->m_token != 0 )
      {
         tok->fromUTF8(tk->m_token);
      }
      else {
         tok->fromUTF8(tk->m_result.begin(), tk->m_result.length() );
      }

      onTokenFound(id, running, tok, tk->m_data);

      if( ownText() )
      {
         delete running;
      }

      if( ownToken() )
      {
         delete tok;
      }

      return true;
   }
   else
   {
      // we already refilled to the best of our capacity -- call onTokenFound with "residual text"

      String* running = new String;
      running->fromUTF8(m_buffer + m_bufPos, m_bufLen - m_bufPos);
      m_bufPos = m_bufLen = 0;
      _p->m_mtx.unlock();

      onTokenFound(-1, running, 0, _p->m_resData);

      if( ownText() )
      {
         delete running;
      }

      return true;
   }

   return false;
}


bool StreamTokenizer::hasNext() const
{
   bool res;

   // if we have a text reader, the thing is over when we hit eof.
   _p->m_mtx.lock();

   if( m_tr != 0 )
   {
      // and of course, when we also have exhausted our buffer.
      res = !( m_bufPos == m_bufLen && m_tr->eof() );
   }
   else {
      res = m_bufPos != m_bufLen;
   }
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
   re2::StringPiece spc(m_buffer+m_bufPos, (int)m_bufLen);

   while( iter != end )
   {
      Token* tk = *iter;

      if( tk->m_token != 0 )
      {
         length_t foundAt = findText( tk->m_token, tk->m_tokenLen );
         if( foundAt < pos )
         {
            pos = foundAt;
            id = tk->m_id;
         }
      }
      else {
         fassert( tk->m_regex != 0 );
         if( tk->m_regex->Match( spc, 0, m_bufLen-m_bufPos, re2::RE2::UNANCHORED, &tk->m_result, 1 ) )
         {
            length_t foundAt = static_cast<length_t>(tk->m_result.begin() - spc.begin())+m_bufPos;
            if( foundAt < pos )
            {
               pos = foundAt;
               id = tk->m_id;
            }
         }
      }
      ++iter;
   }

   return id != -1;
}


length_t StreamTokenizer::findText( const char* token, length_t len )
{
   length_t tpos = 0;
   length_t rest = m_bufLen - m_bufPos;
   char* buf = m_buffer + m_bufPos;

   while( tpos != len && rest >= len )
   {
      if ( buf[tpos] != token[tpos] )
      {
         tpos = 0;
         ++buf;
         --rest;
      }
      else {
         tpos ++;
      }
   }

   if( tpos == len )
   {
      return static_cast<length_t>(buf - m_buffer);
   }

   return String::npos;
}


bool StreamTokenizer::refill()
{
   if( m_tr != 0 )
   {
      // check the invariant
      fassert( m_bufPos <= m_bufLen );

      // We have created bufSize using a *2 factor.
      // the first time, with m_bufLen == 0, read all.
      length_t bs2 = m_bufLen > 0 ? m_bufSize/2 : m_bufSize;
      String str;
      int32 rlen = m_tr->read(str, bs2);

      // hit eof?
      if( rlen == 0 )
      {
         return false;
      }

      // need to move the upper size of the buffers back?
      if ( m_bufPos > bs2 )
      {
         // copy the upper part in the lower part.
         m_bufPos -= bs2;
         m_bufLen -= bs2;
         memcpy(m_buffer, m_buffer +bs2, bs2);
      }

      // now try to utf8-ize the new stuff in the read string.
      length_t len = 0;
      while( (len = str.toUTF8String(m_buffer+m_bufLen, m_bufSize - m_bufLen)) == String::npos )
      {
         // we have to allocate enough memory.
         m_bufSize *= 2;
         char* buf2 = new char[m_bufSize];
         memcpy( buf2, m_buffer, m_bufLen );
         delete[] m_buffer;
         m_buffer = buf2;
      }

      m_bufLen += len;
      return true;
   }

   return false;
}

}

/* end of streamtok.cpp */
