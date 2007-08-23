/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: time_sys_win.cpp
   $Id: time_sys_win.cpp,v 1.2 2007/06/22 15:14:24 jonnymind Exp $

   Win system implementation for time related service
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun mar 6 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Win system implementation for time related service
*/


#include <windows.h>
#include <falcon/sys.h>
#include <falcon/timestamp.h>
#include <falcon/time_sys_win.h>

namespace Falcon {

namespace Sys {
namespace Time {

void currentTime( ::Falcon::TimeStamp &ts )
{
   SYSTEMTIME st;

   GetLocalTime( &st );

   ts.m_year = st.wYear;
   ts.m_month = st.wMonth;
   ts.m_day = st.wDay;
   ts.m_hour = st.wHour;
   ts.m_minute = st.wMinute;
   ts.m_second = st.wSecond;
   ts.m_msec = st.wMilliseconds;
   // todo: collect day of year and weekday
   ts.m_timezone = tz_local;
}


TimeZone getLocalTimeZone()
{
   return tz_local;
}

bool sleep( numeric seconds )
{
   ::Sleep( (DWORD) (seconds * 1000.0) );
   return true;
}

numeric seconds()
{
   return Falcon::Sys::_seconds();
}

bool absoluteWait( const TimeStamp &ts )
{
   TimeStamp now;

   currentTime( now );
   if ( ts <= now )
      return false;

   TimeStamp diff = ts - now;

   DWORD msecs =  diff.m_msec + diff.m_day * (24*3600) + diff.m_hour * 3600 +
      diff.m_minute * 60 + diff.m_second;

   ::Sleep( msecs );
   return true;
}

bool relativeWait( const TimeStamp &ts )
{
   DWORD msecs =  ts.m_msec + ts.m_day * (24*3600) + ts.m_hour * 3600 +
      ts.m_minute * 60 + ts.m_second;

   ::Sleep( msecs );
   return true;
}

bool nanoWait( int32 seconds, int32 nanoseconds )
{
   DWORD msecs =  seconds * 1000 + nanoseconds / 1000000;
   ::Sleep( msecs );
   return true;
}

void timestampFromSystemTime( const SystemTime &sys_time, TimeStamp &ts )
{
   const WinSystemTime *win_time = static_cast< const WinSystemTime *>( &sys_time );

   ts.m_year = win_time->m_time.wYear;
   ts.m_month = win_time->m_time.wMonth;
   ts.m_day = win_time->m_time.wDay;
   ts.m_hour = win_time->m_time.wHour;
   ts.m_minute = win_time->m_time.wMinute;
   ts.m_second = win_time->m_time.wSecond;
   ts.m_msec = win_time->m_time.wMilliseconds;

   // todo: collect day of year and weekday
   ts.m_timezone = tz_local;

}

} // time
} // sys

}


/* end of time_sys_win.cpp */
