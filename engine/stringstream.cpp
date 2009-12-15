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

namespace Falcon {

class StringStream::Buffer {
public:
   byte *m_membuf;
   uint32 m_length;
   uint32 m_allocated;
   int32 m_lastError;
   int32 m_refcount;
   Mutex m_mtx;
   
   Buffer():
      m_membuf(0),
      m_length(0),
      m_lastError(0),
      m_refcount(1)
   {}
};

StringStream::StringStream( int32 size ):
   Stream( t_membuf ),
   m_b( new Buffer ),
   m_pos(0)
{
   m_b->m_length = 0;
   m_pos = 0;
   m_b->m_lastError = 0;
   if ( size == 0 )
      size = 32;

   if ( size > 0 )
   {
      m_b->m_membuf = (byte *) memAlloc( size );
      m_b->m_allocated = size;

      if ( m_b->m_membuf == 0 )
      {
         m_status = t_error;
         m_b->m_lastError = -1;
      }
      else
         status( t_open );
   }
   else {
      m_b->m_membuf = 0;
      m_b->m_allocated = 0;
      status( t_open );
   }
}

StringStream::StringStream( const String &strbuf ):
   Stream( t_membuf ),
   m_b( new Buffer ),
   m_pos(0)
{
   m_b->m_allocated = strbuf.size();
   m_b->m_length = strbuf.size();
   if ( m_b->m_allocated != 0 )
   {
      m_b->m_membuf = (byte *) memAlloc( m_b->m_allocated );

      if ( m_b->m_membuf == 0 )
      {
         status( t_error );
         m_b->m_lastError = -1;
      }
      else {
         memcpy( m_b->m_membuf, strbuf.getRawStorage(), m_b->m_allocated );
         status( t_open );
      }
   }
   else
      m_b->m_membuf = 0;
}

StringStream::StringStream( const StringStream &strbuf ):
   Stream( strbuf ),
   m_pos( strbuf.m_pos )
{
   atomicInc(strbuf.m_b->m_refcount);
   m_b = strbuf.m_b;
}


StringStream::~StringStream()
{
   if ( atomicDec( m_b->m_refcount ) == 0 )
   {
      close();
      delete m_b;
   }
}


void StringStream::setBuffer( const String &source )
{
   m_pos = 0;
   
   m_b->m_mtx.lock();
   m_b->m_membuf = const_cast< byte *>( source.getRawStorage() );
   m_b->m_length = source.size();
   m_b->m_allocated = source.size();
   m_b->m_lastError = 0;
   m_b->m_mtx.unlock();
}


void StringStream::setBuffer( const char* source, int size )
{
   m_pos = 0;
   
   m_b->m_mtx.lock();
   m_b->m_membuf = (byte *) const_cast< char *>( source );
   m_b->m_length = size == -1 ? strlen( source ) : size;
   m_b->m_allocated = m_b->m_length;
   m_b->m_lastError = 0;
   m_b->m_mtx.unlock();
}


bool StringStream::detachBuffer()
{
   m_b->m_mtx.lock();
   if( m_b->m_membuf != 0 ) {
      m_b->m_allocated = 0;
      m_b->m_length = 0;
      m_b->m_membuf = 0;
      m_b->m_mtx.unlock();
      
      status( t_none );
      return true;
   }
   
   m_b->m_mtx.unlock();
   return false;
}

uint32 StringStream::length() const 
{ 
   return m_b->m_length; 
}

uint32 StringStream::allocated() const 
{ 
   return m_b->m_allocated; 
}

byte *StringStream::data() const
{ 
   return m_b->m_membuf; 
}


String *StringStream::closeToString()
{
   String *temp = new String;
   if( closeToString( *temp ) )
      return temp;
   delete temp;
   return 0;
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
   return (int64) m_b->m_lastError; 
}


void StringStream::transfer( StringStream &strbuf )
{
   atomicInc(strbuf.m_b->m_refcount);
   m_b = strbuf.m_b;
}

bool StringStream::errorDescription( String &description ) const
{
   switch( m_b->m_lastError )
   {
      case 0:  description = "None"; return true;
      case -1: description = "Out of Memory"; return true;
   }

   return false;
}

bool StringStream::close()
{
   m_b->m_mtx.lock();
   if( m_b->m_membuf != 0 ) {
      m_b->m_allocated = 0;
      m_b->m_length = 0;
      byte* mem = m_b->m_membuf;
      m_b->m_membuf = 0;
      status( t_none );
      m_b->m_mtx.unlock();
      
      memFree(mem);
      return true;
   }
   
   m_b->m_mtx.unlock();
   return false;
}

int32 StringStream::read( void *buffer, int32 size )
{
   m_b->m_mtx.lock();
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      m_b->m_mtx.unlock();

      return -1;
   }

   if ( m_pos >= m_b->m_length ) {
      m_status = m_status | t_eof;
      m_b->m_mtx.unlock();
      
      return 0;
   }

   int sret = size + m_pos < m_b->m_length ? size : m_b->m_length - m_pos;
   memcpy( buffer, m_b->m_membuf + m_pos, sret );
   m_pos += sret;
   m_lastMoved = sret;
   m_b->m_mtx.unlock();

   return sret;
}

bool StringStream::readString( String &target, uint32 size )
{
   m_b->m_mtx.lock();

   uint32 chr;
   target.size(0);
   target.manipulator( &csh::handler_buffer );

   while( size > 0 && get( chr ) )
   {
      target.append( chr );
      size--;
   }
   m_b->m_mtx.unlock();

   return true;
}


int32 StringStream::write( const void *buffer, int32 size )
{
   m_b->m_mtx.lock();
   
   if ( m_b->m_membuf == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }

   if( size + m_pos > m_b->m_allocated ) {
      int32 alloc = m_b->m_allocated + size + 32;
      byte *buf1 = (byte *) memAlloc( alloc );
      if ( buf1 == 0 )
      {
         m_b->m_mtx.unlock();
         m_b->m_lastError = -1;
         return -1;
      }

      m_b->m_allocated = alloc;
      memcpy( buf1, m_b->m_membuf, m_b->m_length );
      memFree( m_b->m_membuf );
      m_b->m_membuf = buf1;
   }

   memcpy( m_b->m_membuf + m_pos, buffer, size );
   m_pos += size;
   if ( m_pos > m_b->m_length )
      m_b->m_length = m_pos;

   m_lastMoved = size;
   m_b->m_mtx.unlock();

   return size;
}

bool StringStream::writeString( const String &source, uint32 begin, uint32 end )
{
   uint32 charSize = source.manipulator()->charSize();
   uint32 start = begin * charSize;
   uint32 stop = source.size();
   if ( end < stop / charSize )
      stop = end * charSize;

   if ( source.size() > 0 )
   {
      return write( source.getRawStorage()+ start, stop - start ) >= 0;
   }

   return false;
}

bool StringStream::put( uint32 chr )
{
   /** \TODO optimize */
   byte b = (byte) chr;
   return write( &b, 1 ) == 1;
}

bool StringStream::get( uint32 &chr )
{
   /** \TODO optimize */
   if( popBuffer( chr ) )
      return true;

   m_b->m_mtx.lock();
   
   if ( m_b->m_membuf == 0 ) 
   {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if ( m_pos >= m_b->m_length )
   {
      m_b->m_mtx.unlock();
      m_status = m_status | t_eof;
      return false;
   }
   
   chr = m_b->m_membuf[m_pos];
   m_pos++;
   m_lastMoved = 1;
   m_b->m_mtx.unlock();
   
   return true;
}

int64 StringStream::seek( int64 pos, Stream::e_whence w )
{
   m_b->m_mtx.lock();

   if ( m_b->m_membuf == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }

   switch( w ) {
      case Stream::ew_begin: m_pos = (int32) pos; break;
      case Stream::ew_cur: m_pos += (int32) pos; break;
      case Stream::ew_end: m_pos = (int32) (m_b->m_length + pos); break;
   }

   if ( m_pos > m_b->m_length )
   {
      m_pos = m_b->m_length;
      m_status = t_eof;
   }
   else if ( m_pos < 0 )
   {
      m_pos = 0;
      m_status = t_none;
   }
   else
      m_status = t_none;

   m_b->m_mtx.unlock();

   return m_pos;
}

int64 StringStream::tell()
{
   m_b->m_mtx.lock();
   if ( m_b->m_membuf == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }
   
   m_b->m_mtx.unlock();
   return m_pos;
}

bool StringStream::truncate( int64 pos )
{
   m_b->m_mtx.lock();
   if ( m_b->m_membuf == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if ( pos <= 0 )
      m_b->m_length = 0;
   else
      m_b->m_length = (int32) pos;

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
   if ( m_b->m_length == 0 )
   {
      target.size(0);
   }
   else {
      char *data = (char *) memAlloc( m_b->m_length );
      memcpy( data, m_b->m_membuf, m_b->m_length );
      target.adopt( data, m_b->m_length, m_b->m_length );
   }
   m_b->m_mtx.unlock();
}

bool StringStream::closeToString( String &target )
{
   m_b->m_mtx.lock();
   if ( m_b->m_membuf == 0 )
   {
      m_b->m_mtx.unlock();
      return false;
   }
   
   if ( m_b->m_length == 0 ) 
   {
      target.size( 0 );
      m_b->m_mtx.unlock();
      return true;
   }

   target.adopt( (char *) m_b->m_membuf, m_b->m_length, m_b->m_allocated );
   m_b->m_membuf = 0;
   m_b->m_length = 0;
   m_b->m_allocated = 0;
   m_b->m_mtx.unlock();
   
   return true;
}

byte * StringStream::closeToBuffer()
{
   m_b->m_mtx.lock();
   if ( m_b->m_membuf == 0 || m_b->m_length == 0)
   {
      m_b->m_mtx.unlock();
      return 0;
   }

   byte *data = m_b->m_membuf;
   m_b->m_membuf = 0;
   m_b->m_length = 0;
   m_b->m_allocated = 0;
   m_b->m_mtx.unlock();
   
   return data;
}

StringStream *StringStream::clone() const
{
   StringStream *sstr = new StringStream( *this );
   return sstr;
}

}


/* end of file_StringStream.cpp */
