/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_syslog.h

   Logging module -- log channel interface (to syslog/event log)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_LOGCHANNEL_SYSLOG_H
#define FALCON_FEATHERS_LOGCHANNEL_SYSLOG_H

#include "logchannel.h"
#include <falcon/textwriter.h>
#include <falcon/timestamp.h>
#include <falcon/mt.h>

namespace Falcon {
namespace Feathers {

/** Logging for Syslog (POSIX) or Event Logger (MS-Windows).  */
class LogChannelSyslog: public LogChannel
{
public:
   LogChannelSyslog( const String& identity, uint32 facility = 0, int level=LOGLEVEL_ALL );
   LogChannelSyslog( const String& identity, const String &fmt, uint32 facility = 0, int level=LOGLEVEL_ALL );

protected:
   String m_identity;
   uint32 m_facility;

   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual void init();
   virtual bool close();

   virtual ~LogChannelSyslog();

private:
   void* m_sysdata;
};

}
}

#endif

/* end of logchannel_syslog.h */
