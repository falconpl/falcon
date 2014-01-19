/*
   FALCON - The Falcon Programming Language.
   FILE: apache_errhand.h

   Falcon module for Apache 2

   Apache aware error handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_APACHE_ERRHAND_H_
#define _FALCON_WOPI_APACHE_ERRHAND_H_

#include <falcon/error.h>
#include <falcon/string.h>
#include "../include/falcon/wopi/client.h"
#include "../include/falcon/wopi/errorhandler.h"

/** Error handler specialized for the apache module.
   TODO: Add support for configurable error report form.
*/
class ApacheErrorHandler: public Falcon::WOPI::ErrorHandler
{
public:
   ApacheErrorHandler(bool bFancy=true);
   virtual ~ApacheErrorHandler();

   virtual void replyError( Falcon::WOPI::Client* client, const Falcon::String& message );
   virtual void logSysError( Falcon::WOPI::Client* client, int code, const Falcon::String& message );
   virtual void logError( Falcon::WOPI::Client* client, Falcon::Error* error );

private:
   int m_notifyMode;
};

#endif /* _FALCON_WOPI_APACHE_ERRHAND_H_ */

/* end of apache_errhand.h */

