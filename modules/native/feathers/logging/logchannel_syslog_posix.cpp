/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_syslog_posix.cpp

   Logging module -- Posix-specific syslog interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel_syslog_posix.cpp"

#include "logchannel_syslog.h"
#include <falcon/autocstring.h>

#include <syslog.h>

namespace Falcon {
namespace Feathers {


bool LogChannelSyslog::close()
{
   if( ! LogChannel::close() )
   {
      return false;
   }

   closelog();
   return true;
}

void LogChannelSyslog::init()
{
   if ( m_facility == 0 )
   {
      m_facility = LOG_USER;
   }

   AutoCString app( m_identity );
   openlog( app.c_str() , LOG_NDELAY | LOG_PID, m_facility );
}


void LogChannelSyslog::writeLogEntry( const String& entry, LogMessage* pOrigMsg )
{
   int level;

   switch( pOrigMsg->m_level )
   {
      case LOGLEVEL_FATAL: level = LOG_ALERT; break;
      case LOGLEVEL_ERROR:  level = LOG_ERR; break;
      case LOGLEVEL_WARN: level = LOG_WARNING; break;
      case LOGLEVEL_INFO: level = LOG_INFO; break;
      default: level = LOG_DEBUG; break;
   }

   AutoCString msg(entry);
   syslog( m_facility | level, "%s", msg.c_str() );
}

}
}

/* end of logchannel_syslog_posix.cpp */
