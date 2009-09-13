/*
   FALCON - The Falcon Programming Language
   FILE: logging_srv_syslog.cpp

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Sep 2009 16:42:41 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "logging_mod.h"
#include <falcon/error.h>
#include <falcon/autocstring.h>

#include <syslog.h>

namespace Falcon {

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
   stop();
   closelog();
}


void LogChannelSyslog::init()
{
   if ( m_facility == 0 )
      m_facility = LOG_USER;

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

/* end of logging_srv_syslog.cpp */
