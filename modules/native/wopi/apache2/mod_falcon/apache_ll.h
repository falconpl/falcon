/*
   FALCON - The Falcon Programming Language.
   FILE: apache_ll.h

   Apache Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WOPI_APACHELOGLISTNER_H_
#define _FALCON_WOPI_APACHELOGLISTNER_H_

#include <falcon/setup.h>
#include <falcon/log.h>

#include <httpd.h>
#include <http_log.h>

namespace Falcon {
namespace WOPI {

/** App Log Listener for WOPI.
 *
 */
class ApacheLogListener: public Log::Listener
{
public:
   ApacheLogListener( apr_pool_t* pool );
   virtual ~ApacheLogListener();

protected:
   virtual void onMessage( int fac, int lvl, const String& message );

private:
   apr_pool_t* m_pool;
};

}
}

#endif

/* end of apache_ll.h */
