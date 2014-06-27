/*
   FALCON - The Falcon Programming Language.
   FILE: reader.cpp

   Base abstract class for input stream readers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 12:56:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/reader.h>
#include <falcon/sys.h>
#include <falcon/stream.h>
#include <falcon/stderrors.h>

#include <stdio.h>
#include <string.h>


namespace Falcon {

Reader::Reader():
   m_buffer(0),
   m_bufPos(0),
   m_bufLength(0),
   m_bufSize(0),
   m_readSize(0),
   m_stream(0)
{}

Reader::Reader( Stream* stream ):
   m_stream(stream)
{
   m_readSize = Sys::_getPageSize();

   m_bufLength = 0;
   m_bufPos = 0;

   m_bufSize = m_readSize*2;
   m_buffer = new byte[m_bufSize];

   if( m_stream != 0 )
   {
      m_stream->incref();
   }
}

Reader::Reader( const Reader& other )
{
   if (other.m_stream != 0 )
   {
      other.m_stream->incref();
   }
   m_stream = other.m_stream;

   m_readSize = other.m_readSize;

   m_bufLength = other.m_bufLength;
   m_bufPos = other.m_bufPos;

   m_bufSize = other.m_bufSize;
   m_buffer = new byte[m_bufSize];
   memcpy( m_buffer, other.m_buffer, m_bufLength );
}


Reader::~Reader()
{
   delete[] m_buffer;
   if( m_stream != 0 )
   {
      m_stream->decref();
   }
}

void Reader::delegate( Reader& target )
{

   if( target.m_stream != 0 )
   {
      target.m_stream->decref();
   }
   
   delete[] target.m_buffer;

   target.m_stream = this->m_stream;

   target.m_buffer = this->m_buffer;
   target.m_bufPos = this->m_bufPos;
   target.m_bufLength = this->m_bufLength;
   target.m_bufSize = this->m_bufSize;
   target.m_readSize = this->m_readSize;

   m_buffer = 0;
   m_bufLength = 0;
   m_bufPos = 0;
   m_bufSize = 0;
   this->m_stream = 0;

   if( this->m_stream != 0 )
   {
      this->m_stream->incref();
   }
}

void Reader::setBufferSize( length_t bs )
{
   fassert( m_stream != 0 );

   fassert( bs > 0 );

   m_readSize = bs;
   m_bufSize = m_readSize*2;

   byte* nBuf = new byte[m_bufSize];
   if ( m_bufPos < m_bufLength )
   {
      memmove( nBuf, m_buffer, m_bufLength - m_bufPos );
   }

   m_bufLength -= m_bufPos;
   m_bufPos = 0;
   delete[] m_buffer;
   m_buffer = nBuf;
}

void Reader::changeStream( Stream* s, bool bDiscard )
{
   // at times, changeStream is used just to discard the buffer.
   if( s != m_stream )
   {
      if( s != 0 )
      {
         s->incref();
      }

      if( m_stream != 0 )
      {
         m_stream->decref();
      }

      m_stream = s;
   }
   

   if ( bDiscard )
   {
      m_bufPos = 0;
      m_bufLength = 0;
   }
}


void Reader::sync()
{
   m_bufPos = 0;
   m_bufLength = 0;
}


bool Reader::refill()
{
   fassert( m_stream != 0 );
   
   // have we got enough room for a single read?
   if ( m_readSize + m_bufLength > m_bufSize )
   {
      // we must make room for m_suggestedBufferSize bytes -- or give up
      if( m_bufPos + m_readSize > m_bufLength )
      {
         if( m_bufPos < m_bufLength )
         {
            // if there is unused data, copy it back.
            memmove( m_buffer, m_buffer + m_bufPos, m_bufLength - m_bufPos );
         }
         
         m_bufLength -= m_bufPos;
         m_bufPos = 0;
      }
      else
      {
         // there is just too much data to be used to read it again.
         return true;
      }
   }

   size_t nSize = m_stream->read(m_buffer + m_bufLength, m_readSize );
   if ( nSize == (size_t) -1 )
   {
      // if the stream wanted to throw, we wouldn't be here.
      return false;
   }

   m_bufLength += nSize;
   return true;
}


bool Reader::fetch( length_t suggestedSize )
{
   // fast path -- do we have enough data?
   if( m_bufPos + suggestedSize <= m_bufLength )
   {
      return true;
   }

   if ( m_stream->eof() )
   {
      // Is there still something to read?
      return m_bufPos < m_bufLength;
   }

   // no, we don't have enough data. Is our buffer large enough?
   if( m_readSize < suggestedSize )
   {
      long pageSize = Sys::_getPageSize();
      // change the read size to a suitable size.
      length_t nLen = ((suggestedSize/pageSize)+1)*pageSize;
      setBufferSize( nLen );
   }

   // refill just once
   return refill();
}


bool Reader::ensure( length_t size )
{
   // fast path -- do we have enough data?
   if( m_bufPos + size <= m_bufLength )
   {
      return true;
   }

   if ( m_stream->eof() )
   {
      return false;
   }

   // no, we don't have enough data. Is our buffer large enough?
   if( m_readSize < size )
   {
      long pageSize = Sys::_getPageSize();
      // change the read size to a suitable size.
      length_t nLen = ((size/pageSize)+1)*pageSize;
      setBufferSize( nLen );
   }

   while( m_bufPos + size > m_bufLength )
   {
      if ( ! refill() )
      {
         // we know for sure that the stream don't has should throw set,
         // or it would have thrown us out
         return false;
      }

      // did we hit eof?
      if ( m_stream->eof() ) 
      {
         // here we should throw if the stream has this setting.
         if( m_stream->shouldThrow() )
         {
            throw new IOError(ErrorParam(e_deser_eof, __LINE__, __FILE__));
         }

         return false;
      }
   }

   // I don't know if we read more, but we read enough.
   return true;
}


bool Reader::eof() const
{
   fassert( m_stream != 0 );

   return m_bufPos >= m_bufLength && m_stream->eof();
}

}

/* end of reader.cpp */
