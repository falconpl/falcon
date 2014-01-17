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

ApacheErrorHandler::ApacheErrorHandler(int cfgNotifyMode):
   m_notifyMode(cfgNotifyMode)
{}

ApacheErrorHandler::~ApacheErrorHandler()
{}

void ApacheErrorHandler::replyError( Falcon::WOPI::Client* client, int code, const Falcon::String& message )
{
   // TODO: manage encoding. For now, UTF8 is ok
   Falcon::AutoCString cerr( message );

   ApacheRequest* req = static_cast<ApacheRequest*>(client->request());

   // surely, we have to log the error.
   ap_log_rerror( APLOG_MARK, APLOG_WARNING, 0, req->apacheRequest(),
         "Error in script when processing %s:\n%s",
         req->apacheRequest()->filename,
         cerr.c_str()
         );

   // we do different things depending on our level.
   if ( m_notifyMode == FM_ERROR_MODE_SILENT )
   {
      return;
   }

   // should we just tell we had an error?
   client->reply()->status(code);
   client->c
   client->reply()->commit();
   if ( m_notifyMode == FM_ERROR_MODE_KIND )
   {
      const char* msg = "<B>Falcon script error - check the logs.</B>";
      client->stream()->write( msg, strlen(msg) );
   }
   else {
      // or shall we send everything?
      client->stream()->write( cerr.c_str(), cerr.length() );
   }

   client->stream()->flush();
}


void ApacheErrorHandler::handleError( Falcon::WOPI::Client* client, Falcon::Error *error )
{
   // in all the other cases we have to log something.
   Falcon::String serr = error->describe( true );
   replyError( client, 500, serr );
}
