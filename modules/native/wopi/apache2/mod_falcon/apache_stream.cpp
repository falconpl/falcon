/*
   FALCON - The Falcon Programming Language.
   FILE: apache_stream.h

   Falcon module for Apache 2
   Apache specific Falcon stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>

#include <httpd.h>
#include <util_filter.h>
#include <apr_strings.h>


#include "apache_stream.h"
#include "apache_output.h"

ApacheStream::ApacheStream( ApacheOutput *output ):
   Falcon::Stream( Falcon::Stream::t_membuf ),
   m_output( output ),
   m_lastError( 0 )
{}

ApacheStream::ApacheStream( const ApacheStream &other ):
   Falcon::Stream( Falcon::Stream::t_membuf ),
   m_output( other.m_output ),
   m_lastError( other.lastError() )
{
}

Falcon::int32 ApacheStream::write( const void *buffer, Falcon::int32 size )
{
   if( m_output->write( (const char*) buffer, size ) )
   {
      return size;
   }

   m_lastError = m_output->lastError();
   return -1;
}


// Text oriented write
bool ApacheStream::writeString( const Falcon::String &source, Falcon::uint32 begin, Falcon::uint32 end )
{
   Falcon::AutoCString buffer( begin == 0 && end == Falcon::String::npos ?
         source :
         source.subString( begin, end ) );

   if( m_output->write( (const char*) buffer.c_str(), buffer.length() ) )
   {
      return buffer.length();
   }

   m_lastError = m_output->lastError();
   return -1;
}

// char oriented write (used mainly by transcoders)
bool ApacheStream::put( Falcon::uint32 chr )
{
    m_lastError = 0;

   // chr may be an UTF-8 code, so we use Falcon::String facilites to encode it.
   Falcon::String temp;
   temp.append( chr );
   Falcon::AutoCString buffer( temp );

   if( m_output->write( (const char*) buffer.c_str(), buffer.length() ) )
   {
      return true;
   }

   m_lastError = m_output->lastError();
   return false;
}

bool ApacheStream::close()
{

   return true;
}

Falcon::int32 ApacheStream::writeAvailable( int, const Falcon::Sys::SystemData* )
{
   return 1;
}

Falcon::int64 ApacheStream::lastError() const
{
   return m_lastError;
}

bool ApacheStream::get( Falcon::uint32 &chr )
{
   status( Falcon::Stream::t_unsupported );
   return -1;
}


bool ApacheStream::flush()
{
   // Do nothing;
   // flush is so heavy here that we want to do it explicitly on the apache output item.
   return true;
}

ApacheStream *ApacheStream::clone() const
{
   return new ApacheStream( *this );
}

//================================================================
// Input stream
//

ApacheInputStream::ApacheInputStream( request_rec *request ):
      Falcon::Stream( Falcon::Stream::t_membuf ),
      m_lastError(0),
      m_request( request )
{
   m_brigade = apr_brigade_create(
              request->pool,
              request->connection->bucket_alloc);
}

ApacheInputStream::~ApacheInputStream()
{

}


bool ApacheInputStream::close()
{
   return true;
}

Falcon::int32 ApacheInputStream::read( void *buffer, Falcon::int32 size )
{
   apr_size_t len = (apr_size_t) size;

   if( ap_get_brigade(
            m_request->input_filters, m_brigade, AP_MODE_READBYTES, APR_BLOCK_READ, len)
         == APR_SUCCESS)
   {
      apr_brigade_flatten(m_brigade, (char*) buffer, &len);
      apr_brigade_cleanup(m_brigade);
      if( len == 0 )
         status( t_eof );

      m_lastMoved = len;
      return len;
   }

   m_lastError = -1;
   return -1;
}

Falcon::int32 ApacheInputStream::readAvailable( int, const Falcon::Sys::SystemData* )
{
   return 1;
}

Falcon::int64 ApacheInputStream::lastError() const
{
   return m_lastError;
}


bool ApacheInputStream::get( Falcon::uint32 &chr )
{
   // inefficient, but we're using the buffered read.
   apr_size_t len = 1;
   Falcon::byte b;

   if( ap_get_brigade(
            m_request->input_filters, m_brigade, AP_MODE_READBYTES, APR_BLOCK_READ, 1)
         == APR_SUCCESS)
   {
      apr_brigade_flatten(m_brigade, (char*)&b, &len);
      apr_brigade_cleanup(m_brigade);
      if( len == 0 )
         status( t_eof );

      m_lastMoved = len;
      chr = (Falcon::uint32) b;

      return true;
   }


   return false;
}


ApacheInputStream *ApacheInputStream::clone() const
{
   return 0;
}


/* end of apache_stream.h */

