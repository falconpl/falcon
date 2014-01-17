/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_eh.h

   Micro HTTPD server providing Falcon scripts on the web.

   Error handler for falhttpd.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALHTTPD_ERRORHANDLER_H_
#define _FALHTTPD_ERRORHANDLER_H_

#include <falcon/wopi/client.h>
#include <falcon/wopi/errorhandler.h>

namespace Falcon {

/** Error handler for falhttpd.
 *
 */
class SHErrorHandler: public WOPI::ErrorHandler
{
public:
   SHErrorHandler( bool bFancy );
   virtual ~SHErrorHandler();
   virtual void replyError( WOPI::Client* client, const String& message );
   virtual void logSysError( WOPI::Client* client, int code, const String& message );
   virtual void logError( WOPI::Client* client, Error* error );
};

}

#endif /* _FALHTTPD_EH_H_ */

/* falhttpd_eh.h */
