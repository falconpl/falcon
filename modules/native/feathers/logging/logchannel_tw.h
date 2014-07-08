/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_tw.h

   Logging module -- log channel interface (for textwriter)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_LOGCHANNEL_TW_H
#define FALCON_FEATHERS_LOGCHANNEL_TW_H

#include "logchannel.h"

namespace Falcon {

class TextWriter;

namespace Feathers {

/** Logging channel writing text on a TextWriter
 *
 */
class LogChannelTW: public LogChannel
{

public:
   LogChannelTW( TextWriter* s, int level=LOGLEVEL_ALL );
   LogChannelTW( TextWriter* s, const String &fmt, int level=LOGLEVEL_ALL );

   inline bool flushAll() const { return m_bFlushAll; }
   inline void flushAll( bool b ) { m_bFlushAll = b; }

   virtual bool close();

protected:
   TextWriter* m_stream;
   bool m_bFlushAll;
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual ~LogChannelTW();

};

}
}

#endif

/* end of logchannel_tw.h */
