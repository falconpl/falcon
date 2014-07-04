/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_engine.h

   Logging module -- log channel interface (to Falcon engine logging)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_LOGCHANNEL_ENGINE_H
#define FALCON_FEATHERS_LOGCHANNEL_ENGINE_H

#include "logchannel.h"

namespace Falcon {
namespace Feathers {

/** Logging channel writing to the standard Falcon engine log facility
 *
 */
class LogChannelEngine: public LogChannel
{

public:
   LogChannelEngine( int level=LOGLEVEL_ALL );
   LogChannelEngine( const String &fmt, int level=LOGLEVEL_ALL );

   virtual bool close();

protected:
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual ~LogChannelEngine();
};

}
}

#endif

/* end of logchannel_engine.h */
