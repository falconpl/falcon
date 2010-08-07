/*
   FALCON - The Falcon Programming Language
   FILE: logging_srv_win.cpp

   Logging module -- service classes (MS-Windows specific)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "logging_mod.h"
#include <falcon/error.h>
#include <falcon/autowstring.h>

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
   CloseEventLog( (HANDLE) m_sysdata );
}

void LogChannelSyslog::init()
{
   AutoWString appname( m_identity );
   m_sysdata = (void*) OpenEventLogW( NULL, appname.w_str() );
   
   if ( m_sysdata == 0 )
   {
      throw new IoError( ErrorParam( FALCON_LOGGING_ERROR_OPEN, __LINE__ )
		  .origin( e_orig_runtime )
		  .sysError( GetLastError() ) );
   }
}


void LogChannelSyslog::writeLogEntry( const String& entry, LogChannel::LogMessage* pOrigMsg )
{
   WORD wType;
   DWORD dwEventID;

   if ( pOrigMsg->m_level <= LOGLEVEL_ERROR )
   {  
      wType = EVENTLOG_ERROR_TYPE;
	  dwEventID = 3 << 30;
   }
   else if ( pOrigMsg->m_level == LOGLEVEL_WARN )
   {
      wType = EVENTLOG_WARNING_TYPE;
	  dwEventID = 2 << 30;
   }
   else {
      wType = EVENTLOG_INFORMATION_TYPE;
	  dwEventID = 1 << 30;
   }

   // From MS docs; event ID  = gravity | custom | facility | code;
   dwEventID |= (1 << 29) | ((m_facility&0x1FF) << 16) | (pOrigMsg->m_code & 0xFFFF);

   AutoWString w_msg( entry );
   
   const wchar_t* strings[] = { w_msg.w_str() };
   ReportEventW(
	   (HANDLE)m_sysdata,	// __in  HANDLE hEventLog,
	   wType,				// __in  WORD wType,
	   m_facility,			// __in  WORD wCategory,
	   dwEventID,			// __in  DWORD dwEventID,
	   NULL,				// __in  PSID lpUserSid,
	   1,					// __in  WORD wNumStrings,
	   0,					// __in  DWORD dwDataSize,
	   strings,				// __in  LPCTSTR *lpStrings,
	   NULL					// __in  LPVOID lpRawData
	   );

}
	   
}

/* end of logging_srv_win.cpp */
