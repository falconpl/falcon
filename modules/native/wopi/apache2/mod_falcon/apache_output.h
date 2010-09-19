/*
   FALCON - The Falcon Programming Language.
   FILE: apache_output.h

   Falcon module for Apache 2

   Little utilities to write data on the apache logical memory
   output stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

 */

#ifndef MOD_FALCON_OUTPUT_DATA
#define MOD_FALCON_OUTPUT_DATA

#include <falcon/wopi/reply.h>
#include <falcon/string.h>
#include "mod_falcon.h"

class ApacheOutput
{
public:

   ApacheOutput( request_rec *rec );
   ~ApacheOutput()
   {
      close();
   }

   bool write( const char *data, int size );
   bool write( const Falcon::String &data );
   void flush();
   void close();
   apr_status_t lastError() const { return m_lastError; }
   request_rec *request() const { return m_request; }

private:
   request_rec *m_request;
   apr_bucket_brigade *m_brigade;
   bool m_bHeaderSent;
   bool m_bClosed;
   apr_status_t m_lastError;

   // Callback generated on write overflow.
   /* It shall send the first headers if still not done. */
   static apr_status_t s_onWriteOverflow( apr_bucket_brigade *bb, void *ctx );

};

#endif
