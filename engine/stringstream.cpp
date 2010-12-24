/*
   FALCON - The Falcon Programming Language.
   FILE: file_StringStream.cpp

   Implementation of StringStream oriented streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 13 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of StringStream oriented streams.
*/

#include <falcon/stringstream.h>
#include <falcon/memory.h>
#include <falcon/mt.h>
#include <cstring>
#include <string.h>

#include <stdio.h>

namespace Falcon {

class StringStream::Buffer {
public:
   String* m_str;
   int32 m_refcount;
   Mutex m_mtx;
   
   Buffer():
      m_str( new String() ),
      m_refcount(1)
   {}

   Buffer( const Buffer& other ):
      m_str( new String( *other.m_str ) ),
      m_refcount(1)
   {}

   void incref() { atomicInc(m_refcount); }
   void decref() { if (atomicDec(m_refcount) == 0) delete this; }

private:
   ~Buffer() {
      delete m_str;
   }
};

StringStream::StringStream( int32 size ):
   Stream( t_membuf ),
   m_b( new Buffer ),
   m_lastError(0),
   m_pos(0)
{
   m_pos = 0;
   if ( size <= 0 )
      size = 32;

   m_b->m_str->reserve(size);
}

StringStream::StringStream( const String &strbuf ):
   Stream( t_membuf ),
   m_b( new Buffer ),
   m_lastError(0),
   m_pos(0)
{
   *m_b->m_str = strbuf;
   m_b->m_str->bufferize();
}

StringStream::StringStream( const StringStream &strbuf ):
   Stream( strbuf ),
   m_lastError(0),
   m_pos( strbuf.m_pos )
{
   strbuf.m_b->incref();
   m_b = strbuf.m_b;
}


StringStream::~StringStream()
{
   m_b->decref();
}


void StringStream::setBuffer( const String &source )
{
   m_pos = 0;
   
   m_b->m_mtx.lock();
   if( m_b->m_str == 0 )
   {
      m_b->m_str = new String( source );
   }
   else
   {
      *m_b->m_str = source;
   }

   m_b->m_str->bufferize();
   m_lastError = 0;
   m_b->m_mtx.unlock();
}


void StringStream::setBuffer( const char* source, int size )
{
   m_b->m_mtx.lock();
   m_pos = 0;
   if( m_b->m_str == 0 )
   {
      m_b->m_str = new String;
   }
   m_b->m_str->reserve(size);
   memcpy( m_b->m_str->getRawStorage(), source, size );
   m_b->m_str->size( size );
   m_b->m_str->manipulator( &csh::handler_buffer );
   m_lastError = 0;
   m_b->m_mtx.unlock();
}


bool StringStream::detachBuffer()
{
   m_b->m_mtx.lock();
   if( m_b->m_str != 0 ) {
      m_b->m_str->setRawStorage(0);
      m_b->m_str->size(0);
      m_b->m_str->allocated(0);
      m_b->m_str->manipulator( &csh::handler_static );
      m_b->m_mtx.unlock();
      
      status( t_none );
      return true;
   }
   
   m_b->m_mtx.unlock();
   return false;
}

uint32 StringStream::length() const 
{ 
   return m_b->m_str->size();
}

uint32 StringStream::allocated() const 
{ 
   return m_b->m_str->allocated();
}

byte *StringStream::data() const
{ 
   return m_b->m_str->getRawStorage();
}


String *StringStream::closeToString()
{
   m_b->m_mtx.lock();
   String *s = m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();
   return s;
}


CoreString *StringStream::closeToCoreString()
{
   CoreString *temp = new CoreString;
   if (closeToString( *temp ))
      return temp;
   return 0; // CoreString is subject to GC
}


int64 StringStream::lastError() const
{ 
   return (int64) m_lastError;
}


void StringStream::transfer( StringStream &strbuf )
{
   strbuf.m_b->incref();
   m_b->decref();
   m_b = strbuf.m_b;
}

bool StringStream::errorDescription( String &description ) const
{
   switch( m_lastError )
   {
      case 0:  description = "None"; return true;
      case -1: description = "Out of Memory"; return true;
   }

   return false;
}

bool StringStream::close()
{
   m_b->m_mtx.lock();
   delete m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();
   return false;
}

int32 StringStream::read( void *buffer, int32 size )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_status = t_error;
      m_b->m_mtx.unlock();

      return -1;
   }

   uint32 bsize =  m_b->m_str->size();

   if ( m_pos >= bsize ) {
      m_status = m_status | t_eof;
      m_b->m_mtx.unlock();
      
      return 0;
   }

   int sret = size + m_pos < bsize ? size : bsize - m_pos;
   memcpy( buffer, m_b->m_str->getRawStorage() + m_pos, sret );
   m_pos += sret;
   m_lastMoved = sret;
   m_b->m_mtx.unlock();

   return sret;
}

bool StringStream::readString( String &target, uint32 size )
{
   uint32 chr;
   target.size(0);
   target.manipulator( &csh::handler_buffer );

   while ( size > 0 && popBuffer(chr) )
   {
      target += chr;
      --size;
   }

   if( size == 0 )
   {
      return true;
   }

   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   String* src = m_b->m_str;
   int csize = src->manipulator()->charSize();
   int tglen = src->length();
   uint32 start = (m_pos / csize);
   if ( start >= tglen )
   {
      m_status = m_status | t_eof;
      m_b->m_mtx.unlock();
      return false;
   }

   uint32 end = start + size;
   if ( end > tglen ) {
      end = tglen;
   }

   target = src->subString(start,end);
   m_pos = end * csize;

   m_b->m_mtx.unlock();
   return true;
}


int32 StringStream::write( const void *buffer, int32 size )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }
   
   // be sure there is enough space to write
   m_b->m_str->reserve(m_pos+size);

   // effectively write
   memcpy( m_b->m_str->getRawStorage() + m_pos, buffer, size );
   m_pos += size;

   // are we writing at end? -- enlarge the string.
   if( m_pos > m_b->m_str->size() )
   {
      m_b->m_str->size( m_pos );
   }

   m_lastMoved = size;
   m_b->m_mtx.unlock();

   return size;
}

bool StringStream::writeString( const String &source, uint32 begin, uint32 end )
{
   if( begin != 0 || ( end != -1 && end < source.length() ) )
   {
      return StringStream::subWriteString( source.subString(begin, end) );
   }
   else
   {
      return StringStream::subWriteString( source );
   }
}


bool StringStream::subWriteString( const String &source )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }

   // writing at end?
   if ( m_pos >= m_b->m_str->size() )
   {
      // easy
      *m_b->m_str += source;
      m_pos = m_b->m_str->size();
   }
   else {
      // less easy
      int32 origCharSize = m_b->m_str->manipulator()->charSize();
      int32 startPos = m_pos / origCharSize;
      if ( source.length() + startPos < m_b->m_str->length() )
      {
         m_b->m_str->size(m_pos);
         *m_b->m_str += source;
      }
      else
      {
         // even less easy
         String part2 = m_b->m_str->subString( source.length() + startPos );
         m_b->m_str->size(m_pos);
         *m_b->m_str += source + part2;
      }

      // the manipulator may have changed, so pos may have changed as well.
      m_pos = (startPos + source.length()) * m_b->m_str->manipulator()->charSize();
   }

   m_b->m_mtx.unlock();
   return true;
}


bool StringStream::put( uint32 chr )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if( m_pos == m_b->m_str->size() )
   {
      m_b->m_str->append(chr);
      // manipulator may have changed.
      m_pos = m_b->m_str->size();
   }
   else
   {
      int32 pos = m_pos / m_b->m_str->manipulator()->charSize();
      m_b->m_str->setCharAt( pos , chr );
      // manipulator may have changed.
      m_pos = (pos+1) * m_b->m_str->manipulator()->charSize();
   }
   m_b->m_mtx.unlock();

   return true;
}

bool StringStream::get( uint32 &chr )
{
   if( popBuffer( chr ) )
      return true;

   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if( m_pos == m_b->m_str->size() )
   {
      m_status = m_status | t_eof;
      m_b->m_mtx.unlock();
      return false;
   }
   else
   {
      int32 cs = m_b->m_str->manipulator()->charSize();
      chr = m_b->m_str->getCharAt( m_pos/cs );
      // manipulator may have changed.
      m_pos += cs;
   }
   m_b->m_mtx.unlock();
   return true;
}

int64 StringStream::seek( int64 pos, Stream::e_whence w )
{
   m_b->m_mtx.lock();

   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }

   switch( w ) {
      case Stream::ew_begin: m_pos = (int32) pos; break;
      case Stream::ew_cur: m_pos += (int32) pos; break;
      case Stream::ew_end: m_pos = (int32) (m_b->m_str->size() + (pos-1)); break;
   }

   if ( m_pos > m_b->m_str->size() )
   {
      m_pos = m_b->m_str->size();
      m_status = t_eof;
   }
   else if ( m_pos < 0 )
   {
      m_pos = 0;
      m_status = t_none;
   }
   else
      m_status = t_none;

   pos = m_pos;
   m_b->m_mtx.unlock();

   return pos;
}

int64 StringStream::tell()
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }
   
   int32 pos = m_pos;
   m_b->m_mtx.unlock();
   return pos;
}

bool StringStream::truncate( int64 pos )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if ( pos <= 0 )
      m_b->m_str->size( 0 );
   else
      m_b->m_str->size( (int32) pos );

   m_b->m_mtx.unlock();
   return true;
}

int32 StringStream::readAvailable( int32, const Sys::SystemData * )
{
   return 1;
}

int32 StringStream::writeAvailable( int32, const Sys::SystemData * )
{
   return 1;
}

void StringStream::getString( String &target ) const
{
   m_b->m_mtx.lock();
   // as the string is always buffered, this operation is safe
   target = *m_b->m_str;
   m_b->m_mtx.unlock();
}

bool StringStream::closeToString( String &target )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 )
   {
      m_b->m_mtx.unlock();
      return false;
   }
   
   // adopt original string buffer.
   target.adopt( (char*)m_b->m_str->getRawStorage(), m_b->m_str->size(), m_b->m_str->allocated() );
   target.manipulator( (csh::Base*) m_b->m_str->manipulator() ); // apply correct manipulator


   // dispose the old string without disposing of its memory
   m_b->m_str->allocated( 0 );
   m_b->m_str->setRawStorage( 0 );
   delete m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();

   return true;
}

byte * StringStream::closeToBuffer()
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 || m_b->m_str->size() == 0)
   {
      m_b->m_mtx.unlock();
      return 0;
   }
   byte *retbuf = m_b->m_str->getRawStorage();

   // dispose the old string without disposing of its memory
   m_b->m_str->allocated( 0 );
   m_b->m_str->setRawStorage( 0 );
   delete m_b->m_str;
   m_b->m_str = 0;

   m_b->m_mtx.unlock();
   
   return retbuf;
}

StringStream *StringStream::clone() const
{
   StringStream *sstr = new StringStream( *this );
   return sstr;
}

}


/* end of file_StringStream.cpp */
