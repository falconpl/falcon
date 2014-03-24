/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_errhand.cpp

   Falcon module for Apache 2

   Apache aware error handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:17:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/string.h>
#include <falcon/autocstring.h>

#include <httpd.h>
#include <http_log.h>

#include "mod_falcon.h"
#include "apache_errhand.h"
#include "mod_falcon_config.h"
#include "apache_request.h"
#include "apache_reply.h"
#include "apache_ll.h"

ApacheErrorHandler::ApacheErrorHandler( bool bFancy ):
   ErrorHandler(bFancy)
{}

ApacheErrorHandler::~ApacheErrorHandler()
{}


void ApacheErrorHandler::replyError( Falcon::WOPI::Client* client, const Falcon::String& message )
{
   // should we just tell we had an error?
   client->reply()->commit();
   // or shall we send everything?
   Falcon::AutoCString cerr(message);
   client->stream()->write( cerr.c_str(), cerr.length() );
   client->stream()->flush();
}


void ApacheErrorHandler::logSysError( Falcon::WOPI::Client* client, int code, const Falcon::String& message )
{
   Falcon::AutoCString cerr( message );

   ApacheRequest* req = static_cast<ApacheRequest*>(client->request());

   // surely, we have to log the error.
   ap_log_rerror( APLOG_MARK, APLOG_WARNING, 0, req->apacheRequest(),
         "Engine error while processing %s: %d - %s",
         req->apacheRequest()->filename,
         code,
         cerr.c_str()
         );
}


void ApacheErrorHandler::logError( Falcon::WOPI::Client* client, Falcon::Error* error )
{
   Falcon::AutoCString cerr( error->describe(true) );

   ApacheRequest* req = static_cast<ApacheRequest*>(client->request());

   // surely, we have to log the error.
   ap_log_rerror( APLOG_MARK, APLOG_WARNING, 0, req->apacheRequest(),
         "Script error while processing %s:\n %s",
         req->apacheRequest()->filename,
         cerr.c_str()
         );
}

/* end of apache_error.cpp */
