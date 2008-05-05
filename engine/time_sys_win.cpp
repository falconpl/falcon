/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: time_sys_win.cpp

   Win system implementation for time related service
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun mar 6 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

static TimeZone s_cached_timezone = tz_local; // which is also 0

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
   // this function is not reentrant, but it's ok, as
   // the worst thing that may happen in MT or multiprocess
   // is double calculation of the cached value.
      
   // infer timezone by checking gmtime/localtime
   if( s_cached_timezone == tz_local )
   {
      TIME_ZONE_INFORMATION timezone;
      
      DWORD dwMode = GetTimeZoneInformation( &timezone );

      long shift = - (long) ( timezone.Bias );
      if ( dwMode == TIME_ZONE_ID_DAYLIGHT )
         shift -= (long) timezone.DaylightBias;

      long hours = shift / 60;
      long minutes = shift % 60;

      // and now, let's see if we have one of our zones:
      switch( hours )
      {
         case 1: s_cached_timezone = tz_UTC_E_1; break;
         case 2: s_cached_timezone = tz_UTC_E_2; break;
         case 3: s_cached_timezone = tz_UTC_E_3; break;
         case 4: s_cached_timezone = tz_UTC_E_4; break;
         case 5: s_cached_timezone = tz_UTC_E_5; break;
         case 6: s_cached_timezone = tz_UTC_E_6; break;
         case 7: s_cached_timezone = tz_UTC_E_7; break;
         case 8: s_cached_timezone = tz_UTC_E_8; break;
         case 9:
            if( minutes == 30 )
                  s_cached_timezone = tz_ACST;
               else
                  s_cached_timezone = tz_UTC_E_9;
            break;

         case 10:
            if( minutes == 30 )
               s_cached_timezone = tz_ACDT;
            else
               s_cached_timezone = tz_UTC_E_10;
         break;

         case 11:
            if( minutes == 30 )
               s_cached_timezone = tz_NFT;
            else
               s_cached_timezone = tz_UTC_E_11;
         break;

         case 12: s_cached_timezone = tz_UTC_E_11; break;

         case -1: s_cached_timezone = tz_UTC_W_1;
         case -2:
            if( minutes == 30 )
               s_cached_timezone = tz_HAT;
            else
               s_cached_timezone = tz_UTC_W_2;
            break;
         case -3:
            if( minutes == 30 )
               s_cached_timezone = tz_NST;
            else
               s_cached_timezone = tz_UTC_W_3;
            break;
         case -4: s_cached_timezone = tz_UTC_W_4; break;
         case -5: s_cached_timezone = tz_UTC_W_5; break;
         case -6: s_cached_timezone = tz_UTC_W_6; break;
         case -7: s_cached_timezone = tz_UTC_W_7; break;
         case -8: s_cached_timezone = tz_UTC_W_8; break;
         case -9: s_cached_timezone = tz_UTC_W_9; break;
         case -10: s_cached_timezone = tz_UTC_W_10; break;
         case -11: s_cached_timezone = tz_UTC_W_11; break;
         case -12: s_cached_timezone = tz_UTC_W_12; break;

         default:
            s_cached_timezone = tz_NONE;
      }
   }

   return s_cached_timezone;


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
