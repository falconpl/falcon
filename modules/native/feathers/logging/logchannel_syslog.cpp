/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_syslog.cpp

   Logging module -- log channel interface (to syslog/event log)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel_syslog.cpp"

#include "logchannel_syslog.h"

namespace Falcon {
namespace Feathers {


LogChannelSyslog::LogChannelSyslog( const String& identity, uint32 facility, int level ):
   LogChannel( level ),
   m_identity( identity ),
   m_facility( facility )
{
   init();
}

LogChannelSyslog::LogChannelSyslog( const String& identity, const String &fmt, uint32 facility, int level ):
   LogChannel( fmt, level ),
   m_identity( identity ),
   m_facility( facility )
{
   init();
}

LogChannelSyslog::~LogChannelSyslog()
{
   close();
}


}
}

/* end of logchannel_syslog.cpp */
