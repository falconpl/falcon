/*
   FALCON - The Falcon Programming Language.
   FILE: apploglistener.h

   App Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WOPI_APPLOGLISTNER_H_
#define _FALCON_WOPI_APPLOGLISTNER_H_

#include <falcon/setup.h>
#include <falcon/log.h>

namespace Falcon {
namespace WOPI {

/** App Log Listener for WOPI.
 *
 */
class AppLogListener: public Log::Listener
{
public:
   AppLogListener();
   virtual ~AppLogListener();

protected:
   virtual void onMessage( int fac, int lvl, const String& message );
};

}
}

#endif

/* end of apploglistener.h */
