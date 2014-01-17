/*
   FALCON - The Falcon Programming Language.
   FILE: apache_output.cpp

   Falcon module for Apache 2

   Apache aware error handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

 */

#include <http_protocol.h>
#include <util_filter.h>

#include <falcon/autocstring.h>

#include "apache_output.h"

ApacheOutput::ApacheOutput( request_rec *rec ):
   m_request( rec ),
   m_bHeaderSent( false ),
   m_bClosed( false ),
   m_lastError( 0 )
{
   // create the bucket brigade we may need
   m_brigade = apr_brigade_create(rec->pool, rec->connection->bucket_alloc);
}

bool ApacheOutput::write( const char *data, int size )
{
   if ( m_bClosed )
      return false;

   m_lastError =  apr_brigade_write( m_brigade,
      s_onWriteOverflow,
      this,
      data,
      size
   );

   return m_lastError == 0;
}

bool ApacheOutput::write( const Falcon::String &data )
{
   Falcon::AutoCString cstr( data );
   return write( cstr.c_str(), cstr.length() );
}

void ApacheOutput::flush()
{
   /*
   if ( !m_bClosed )
   {
      // Should we write headers first?
      ap_rflush( request() );
      ap_filter_flush( m_brigade, request()->output_filters );
   }
   */
}

void ApacheOutput::close()
{
   flush();
   ap_rflush( request() );
   ap_filter_flush( m_brigade, request()->output_filters );
   apr_brigade_cleanup( m_brigade );
   m_bClosed = true;
}



