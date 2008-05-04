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
#include <cstring>

namespace Falcon {

StringStream::StringStream( int32 size ):
   Stream( t_membuf ),
   m_allocated( size ),
   m_length( 0 ),
   m_pos( 0 ),
   m_lastError( 0 )
{
   if ( size == 0 )
      size = 32;

   if ( size > 0 )
   {
      m_membuf = (byte *) memAlloc( size );
      m_allocated = size;

      if ( m_membuf == 0 )
      {
         m_status = t_error;
         m_lastError = -1;
      }
      else
         status( t_open );
   }
   else {
      m_membuf = 0;
      m_allocated = 0;
      status( t_open );
   }


};

StringStream::StringStream( const String &strbuf ):
   Stream( t_membuf ),
   m_pos( 0 )
{
   m_allocated = strbuf.size();
   m_length = strbuf.size();
   m_membuf = (byte *) memAlloc( m_allocated );

   if ( m_membuf == 0 )
   {
      status( t_error );
      m_lastError = -1;
   }
   else {
      memcpy( m_membuf, strbuf.getRawStorage(), m_allocated );
      status( t_open );
   }
}

StringStream::StringStream( const StringStream &strbuf ):
   Stream( strbuf )
{
   m_length = strbuf.m_length;
   m_pos = strbuf.m_pos;
   m_lastError = strbuf.m_lastError;


   m_allocated = strbuf.m_allocated;

   if ( m_allocated == 0 )
      m_allocated = 32;

   m_membuf = (byte *) memAlloc( m_allocated );

   if ( m_membuf == 0 )
   {
      m_status = t_error;
      m_allocated = 0;
      m_lastError = -1;
   }
   else
      status( t_open );

   memcpy( m_membuf, strbuf.m_membuf, m_length );
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
   if( m_membuf != 0 ) {
      m_allocated = 0;
      m_length = 0;
      memFree(m_membuf);
      m_membuf = 0;
      status( t_none );
      return true;
   }
   return false;
}

int32 StringStream::read( void *buffer, int32 size )
{
   if ( m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   if ( m_pos == m_length ) {
      m_status = m_status | t_eof;
      return 0;
   }

   int sret = size + m_pos < m_length ? size : m_length - m_pos;
   memcpy( buffer, m_membuf + m_pos, sret );
   m_pos += sret;
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
         m_lastError = -1;
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
   if ( m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   if( size + m_pos > m_allocated ) {
      int32 alloc = m_allocated + size + 32;
      byte *buf1 = (byte *) memAlloc( alloc );
      if ( buf1 == 0 )
      {
         m_lastError = -1;
         return -1;
      }

      m_allocated = alloc;
      memcpy( buf1, m_membuf, m_length );
      memFree( m_membuf );
      m_membuf = buf1;
   }

   memcpy( m_membuf + m_pos, buffer, size );
   m_pos += size;
   if ( m_pos > m_length )
      m_length = m_pos;

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

   byte b;
   if ( read( &b, 1 ) == 1 )
   {
      chr = (uint32) b;
      return true;
   }
   return false;
}

int64 StringStream::seek( int64 pos, Stream::e_whence w )
{
   if ( m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   switch( w ) {
      case Stream::ew_begin: m_pos = (int32) pos; break;
      case Stream::ew_cur: m_pos += (int32) pos; break;
      case Stream::ew_end: m_pos = (int32) (m_length + pos); break;
   }

   if ( m_pos > m_length )
      m_pos = m_length;
   else if ( m_pos < 0 )
      m_pos = 0;

   return m_pos;
}

int64 StringStream::tell()
{
   if ( m_membuf == 0 ) {
      m_status = t_error;
      return -1;
   }

   return m_pos;
}

bool StringStream::truncate( int64 pos )
{
   if ( m_membuf == 0 ) {
      m_status = t_error;
      return false;
   }

   if ( pos <= 0 )
      m_length = 0;
   else
      m_length = (int32) pos;

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

String *StringStream::getString() const
{
   if ( m_length == 0 )
      return new String();

   char *data = (char *) memAlloc( m_length );
   memcpy( data, m_membuf, m_length );
   String *ret = new String;
   ret->adopt( data, m_length, m_length );
   return ret;
}

String *StringStream::closeToString()
{
   if ( m_membuf == 0 )
      return 0;

   if ( m_length == 0 )
      return new String();

   String *ret = new String;
   ret->adopt( (char *) m_membuf, m_length, m_allocated );
   m_membuf = 0;
   m_length = 0;
   m_allocated = 0;

   return ret;
}

bool StringStream::closeToString( String &target )
{
   if ( m_membuf == 0 )
      return false;

   if ( m_length == 0 ) {
      target.size( 0 );
      return true;
   }

   target.adopt( (char *) m_membuf, m_length, m_allocated );

   m_membuf = 0;
   m_length = 0;
   m_allocated = 0;
   return true;
}

byte * StringStream::closeToBuffer()
{
   if ( m_membuf == 0 || m_length == 0)
      return 0;

   byte *data = m_membuf;

   m_membuf = 0;
   m_length = 0;
   m_allocated = 0;
   return data;
}

UserData *StringStream::clone() const
{
   StringStream *sstr = new StringStream( *this );
   return sstr;
}

}


/* end of file_StringStream.cpp */
