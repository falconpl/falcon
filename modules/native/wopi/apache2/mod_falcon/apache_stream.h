/*
   FALCON - The Falcon Programming Language.
   FILE: apache_stream.h

   Falcon module for Apache 2
   Apache specific Falcon Stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef APACHE_STREAM_H
#define APACHE_STREAM_H

#include <falcon/stream.h>
#include <falcon/atomic.h>
#include <http_protocol.h>
#include <util_filter.h>

#include <httpd.h>
#include <util_filter.h>
#include <apr_strings.h>
#include <apr_buckets.h>


// Class translating falcon output requests to apache writes.
class ApacheStream: public Falcon::Stream
{
public:

   ApacheStream(request_rec *rec);
   virtual ~ApacheStream();

   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );
   virtual bool close();
   virtual Falcon::int64 tell();
   virtual bool truncate( off_t pos=-1 );
   virtual off_t seek( off_t pos, e_whence w );
   virtual const Falcon::Multiplex::Factory* multiplexFactory() const;
   virtual Stream *clone() const;

   bool isClosed() const { return Falcon::atomicFetch(m_isClosed) == 1; }

private:
   apr_bucket_brigade *m_inBrigade;
   apr_bucket_brigade *m_outBrigade;
   static apr_status_t s_onWriteOverflow( apr_bucket_brigade *bb, void *ctx );

   request_rec *m_req;
   Falcon::atomic_int m_isClosed;
   Falcon::atomic_int m_isHeaderSent;
};

#endif

/* end of apache_stream.h */

