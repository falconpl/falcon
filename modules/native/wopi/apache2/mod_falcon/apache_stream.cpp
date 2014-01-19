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

#include <falcon/stderrors.h>

#include <httpd.h>
#include <util_filter.h>
#include <apr_strings.h>

#include "apache_stream.h"

using namespace Falcon;

ApacheStream::ApacheStream(request_rec *rec)
{
   m_bPS = true;
   m_req = rec;
   m_isClosed = 0;
   m_isHeaderSent = 0;

   // create the bucket brigade we may need
   m_inBrigade = apr_brigade_create(
            rec->pool,
            rec->connection->bucket_alloc);

   m_outBrigade = apr_brigade_create(rec->pool, rec->connection->bucket_alloc);
}

ApacheStream::~ApacheStream()
{
}


size_t ApacheStream::read( void *buffer, size_t size )
{
   apr_size_t len = (apr_size_t) size;

   apr_status_t result = ap_get_brigade(m_req->input_filters, m_inBrigade, AP_MODE_READBYTES, APR_BLOCK_READ, len);
   if( result == APR_SUCCESS)
   {
      apr_brigade_flatten(m_inBrigade, (char*) buffer, &len);
      apr_brigade_cleanup(m_inBrigade);
      if( len == 0 )
      {
         status( t_eof );
      }

      return len;
   }

   if( shouldThrow() )
   {
      throw FALCON_SIGN_XERROR( IOError, e_io_read,
               .extra( String("Reading from Apache Stream: ").N(result)) );
   }

   return -1;
}


size_t ApacheStream::write( const void *buffer, size_t size )
{
   if ( m_isClosed )
      return 0;

   apr_status_t status = apr_brigade_write( m_outBrigade,
      s_onWriteOverflow,
      this,
      static_cast<const char*>(buffer),
      size
   );

   if( status != APR_SUCCESS )
   {
      if( shouldThrow() )
      {
         throw FALCON_SIGN_XERROR( IOError, e_io_write,
                  .extra(String("Writing to Apache Stream: ").N(status)) );
      }

      return -1;
   }

   return size;
}


bool ApacheStream::close()
{
   if( atomicCAS(m_isClosed, 0, 1) )
   {
      flush();
      ap_rflush( m_req );
      ap_filter_flush( m_outBrigade, m_req->output_filters );
      apr_brigade_cleanup( m_outBrigade );
      apr_brigade_cleanup( m_inBrigade );
      return true;
   }
   return false;
}


Falcon::int64 ApacheStream::tell()
{
   throwUnsupported();
   return -1;
}


bool ApacheStream::truncate( Falcon::off_t )
{
   throwUnsupported();
   return false;
}


Falcon::off_t ApacheStream::seek( Falcon::off_t, Falcon::Stream::e_whence )
{
   throwUnsupported();
   return -1;
}


const Falcon::Multiplex::Factory* ApacheStream::multiplexFactory() const
{
   // not multiplexable -- nor exported to scripts.
   return 0;
}


Stream *ApacheStream::clone() const
{
   // not cloneable -- nor exported to scripts.
   return 0;
}


apr_status_t ApacheStream::s_onWriteOverflow( apr_bucket_brigade *, void *ctx )
{
   ApacheStream* stream = static_cast<ApacheStream*>(ctx);

   // first time here?
   if( atomicCAS(stream->m_isHeaderSent, 0, 1 ) )
   {
      ap_rflush( stream->m_req );
   }

   // perform real overwlow flushing
   apr_status_t status = ap_filter_flush( stream->m_outBrigade, stream->m_req->output_filters );

   return status;
}


/* end of apache_stream.h */

