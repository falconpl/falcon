/*
   FALCON - The Falcon Programming Language.
   FILE: webloglistener.h

   Web Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WOPI_WEBLOGLISTNER_H_
#define _FALCON_WOPI_WEBLOGLISTNER_H_

#include <falcon/setup.h>
#include <falcon/log.h>
#include <falcon/textwriter.h>

namespace Falcon {
namespace WOPI {

/** Web Log Listener for WOPI.
 *
 */
class WebLogListener: public Log::Listener
{
public:
   WebLogListener();
   virtual ~WebLogListener();
   void renderLogs( TextWriter* target );

   bool hasLogs() const;

protected:
   class Private;
   Private* _p;
   virtual void onMessage( int fac, int lvl, const String& message );
};

}
}

#endif

/* end of webloglistener.h */
