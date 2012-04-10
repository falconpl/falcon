/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_error.cpp
   $Id$

   Falcon module for Apache 2

   Internal and pre-vm operations error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

//=============================================
// Simple write doc function:
//
#include <http_protocol.h>

struct s_hwritten {
   request_rec *rec;
   int bWritten;
};

static apr_status_t s_errordoc_sendHeaders( apr_bucket_brigade *bb, void *ctx )
{
   struct s_hwritten *shw = (struct s_hwritten*) ctx;
   request_rec *request = shw->rec;
   shw->bWritten = 1;
   apr_table_t *h = request->headers_out;

   apr_table_set( h, "Content-type", "text/html;charset=utf-8" );
   //apr_table_set( h, "Pragma", "no-cache");
   apr_table_set( h, "Cache-Control", "no-cache");
   request->content_type = "text/html;charset=utf-8";

   return APR_SUCCESS;
}

void falcon_mod_write_errorstring( request_rec *rec, const char *str )
{
   struct s_hwritten shw;
   shw.rec = rec;
   shw.bWritten = 0;

   apr_bucket_brigade *brigade =
      apr_brigade_create(rec->pool, rec->connection->bucket_alloc);

   apr_brigade_write( brigade,
      s_errordoc_sendHeaders,
      &shw,
      str,
      strlen( str )
   );

   if ( ! shw.bWritten )
      s_errordoc_sendHeaders( brigade, &shw );

   ap_pass_brigade( rec->output_filters, brigade );
   ap_rflush( rec );
   apr_brigade_cleanup( brigade );
}

