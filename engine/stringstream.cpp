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

StringStream::StringStream( int32 size ):
   Stream( t_membuf ),
   m_b( new Buffer )
{
   m_b->m_length = 0;
   m_b->m_pos = 0;
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
   m_b( new Buffer )
{
   m_b->m_allocated = strbuf.size();
   m_b->m_length = strbuf.size();
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

StringStream::StringStream( const StringStream &strbuf ):
   Stream( strbuf )
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
   if( m_b->m_membuf != 0 ) {
      m_b->m_allocated = 0;
      m_b->m_length = 0;
      memFree(m_b->m_membuf);
      m_b->m_membuf = 0;
      status( t_none );
      return true;
   }
   return false;
}

int32 StringStream::read( void *buffer, int32 size )
{
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   if ( m_b->m_pos == m_b->m_length ) {
      m_status = m_status | t_eof;
      return 0;
   }

   int sret = size + m_b->m_pos < m_b->m_length ? size : m_b->m_length - m_b->m_pos;
   memcpy( buffer, m_b->m_membuf + m_b->m_pos, sret );
   m_b->m_pos += sret;
   m_lastMoved = sret;

   return sret;
}

bool StringStream::readString( String &target, uint32 size )
{
   // TODO Optimize
   uint32 chr;
   target.size(0);
   target.manipulator( &csh::handler_buffer );

   while( size > 0 && get( chr ) )
   {
      target.append( chr );
      size--;
   }

   return true;
   /*
   byte *target_buffer;

   if ( target.allocated() >= size )
      target_buffer = target.getRawStorage();
   else {
      target_buffer = (byte *) memAlloc( size );
      if ( target_buffer == 0 )
      {
         m_b->m_lastError = -1;
         return false;
      }
   }

   int32 sret = this->read( target_buffer, size );
   if ( sret >= 0 )
   {
      target.adopt( (char *) target_buffer, sret );
   }

   return sret;
   */
}

int32 StringStream::write( const void *buffer, int32 size )
{
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   if( size + m_b->m_pos > m_b->m_allocated ) {
      int32 alloc = m_b->m_allocated + size + 32;
      byte *buf1 = (byte *) memAlloc( alloc );
      if ( buf1 == 0 )
      {
         m_b->m_lastError = -1;
         return -1;
      }

      m_b->m_allocated = alloc;
      memcpy( buf1, m_b->m_membuf, m_b->m_length );
      memFree( m_b->m_membuf );
      m_b->m_membuf = buf1;
   }

   memcpy( m_b->m_membuf + m_b->m_pos, buffer, size );
   m_b->m_pos += size;
   if ( m_b->m_pos > m_b->m_length )
      m_b->m_length = m_b->m_pos;

   m_lastMoved = size;

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

   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return false;
   }

   if ( m_b->m_pos == m_b->m_length ) {
      m_status = m_status | t_eof;
      return false;
   }
   
   chr = m_b->m_membuf[m_b->m_pos];
   m_b->m_pos++;
   m_lastMoved = 1;
   return true;
}

int64 StringStream::seek( int64 pos, Stream::e_whence w )
{
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   switch( w ) {
      case Stream::ew_begin: m_b->m_pos = (int32) pos; break;
      case Stream::ew_cur: m_b->m_pos += (int32) pos; break;
      case Stream::ew_end: m_b->m_pos = (int32) (m_b->m_length + pos); break;
   }

   if ( m_b->m_pos > m_b->m_length )
      m_b->m_pos = m_b->m_length;
   else if ( m_b->m_pos < 0 )
      m_b->m_pos = 0;

   return m_b->m_pos;
}

int64 StringStream::tell()
{
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   return m_b->m_pos;
}

bool StringStream::truncate( int64 pos )
{
   if ( m_b->m_membuf == 0 ) {
      m_status = t_error;
      return false;
   }

   if ( pos <= 0 )
      m_b->m_length = 0;
   else
      m_b->m_length = (int32) pos;

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
   if ( m_b->m_length == 0 )
   {
      target.size(0);
   }
   else {
      char *data = (char *) memAlloc( m_b->m_length );
      memcpy( data, m_b->m_membuf, m_b->m_length );
      target.adopt( data, m_b->m_length, m_b->m_length );
   }
}

bool StringStream::closeToString( String &target )
{
   if ( m_b->m_membuf == 0 )
      return false;

   if ( m_b->m_length == 0 ) {
      target.size( 0 );
      return true;
   }

   target.adopt( (char *) m_b->m_membuf, m_b->m_length, m_b->m_allocated );

   m_b->m_membuf = 0;
   m_b->m_length = 0;
   m_b->m_allocated = 0;
   return true;
}

byte * StringStream::closeToBuffer()
{
   if ( m_b->m_membuf == 0 || m_b->m_length == 0)
      return 0;

   byte *data = m_b->m_membuf;

   m_b->m_membuf = 0;
   m_b->m_length = 0;
   m_b->m_allocated = 0;
   return data;
}

FalconData *StringStream::clone() const
{
   StringStream *sstr = new StringStream( *this );
   return sstr;
}

}


/* end of file_StringStream.cpp */
