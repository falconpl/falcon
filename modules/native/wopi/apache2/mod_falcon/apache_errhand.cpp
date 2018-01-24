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

void ApacheErrorHandler::handleError( Falcon::Error *error )
{
   APLOG_USE_MODULE(mod_falcon);

   // we do different things depending on our level.
   if ( m_notifyMode == FM_ERROR_MODE_SILENT )
      return;

   // in all the other cases we have to log something.
   Falcon::String serr;
   error->toString( serr );

   // TODO: manage encoding. For now, UTF8 is ok
   Falcon::AutoCString cerr( serr );
   // surely, we have to log the error.
   ap_log_rerror( APLOG_MARK, APLOG_WARNING, 0, m_output->request(),
         "Error in script when processing %s:\n%s",
         m_output->request()->filename,
         cerr.c_str()
         );

   // should we just tell we had an error?
   if ( m_notifyMode == FM_ERROR_MODE_KIND )
   {
      m_output->write( "<B>Script error - check the logs.</B>" );
   }
   else {
      // or shall we send everything?
      m_output->write( cerr.c_str(), cerr.length() );
   }
}
