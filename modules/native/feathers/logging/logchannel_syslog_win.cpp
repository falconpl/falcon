/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_syslog_win.cpp

   Logging module -- MS-Windows specific syslog interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel_syslog_win.cpp"


#include "logchannel_syslog.h"
#include "logging_mod.h"
#include "logging_fm.h"
#include <falcon/autowstring.h>
#include <falcon/stderrors.h>


#include <windows.h>

namespace Falcon {
namespace Feathers {

bool LogChannelSyslog::close()
{
   if( ! LogChannel::close() )
   {
      return false;
   }

   CloseEventLog( (HANDLE) m_sysdata );
   return true;
}

void LogChannelSyslog::init()
{
   AutoWString appname( m_identity );
   m_sysdata = (void*) OpenEventLogW( NULL, appname.w_str() );

   if ( m_sysdata == 0 )
   {
      throw new IOError( ErrorParam( FALCON_LOGGING_ERROR_OPEN, __LINE__, SRC )
        .desc(FALCON_LOGGING_ERROR_DESC)
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
      (HANDLE)m_sysdata,   // __in  HANDLE hEventLog,
      wType,            // __in  WORD wType,
      m_facility,       // __in  WORD wCategory,
      dwEventID,        // __in  DWORD dwEventID,
      NULL,          // __in  PSID lpUserSid,
      1,             // __in  WORD wNumStrings,
      0,             // __in  DWORD dwDataSize,
      strings,          // __in  LPCTSTR *lpStrings,
      NULL              // __in  LPVOID lpRawData
      );

}

}
}

/* end of logchannel_syslog_win.cpp */
