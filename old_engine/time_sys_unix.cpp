/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_unix.cpp

   Unix system implementation for time related service
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun mar 6 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix system implementation for time related service
*/

#define _REENTRANT
#include <sys/time.h>
#include <falcon/timestamp.h>
#include <falcon/time_sys_unix.h>

namespace Falcon {

namespace Sys {
namespace Time {

static TimeZone s_cached_timezone = tz_local; // which is also 0

void currentTime( ::Falcon::TimeStamp &ts )
{
   struct timeval current;
   struct tm date;
   time_t t;

   time( &t );
   localtime_r( &t, &date );
   ts.m_year = date.tm_year + 1900;
   ts.m_month = date.tm_mon+1;
   ts.m_day = date.tm_mday;
   ts.m_hour = date.tm_hour;
   ts.m_minute = date.tm_min;
   ts.m_second = date.tm_sec;
   // todo: collect day of year and weekday
   ts.m_timezone = tz_local;

   gettimeofday( &current, 0 );
   ts.m_msec = current.tv_usec / 1000;
}


TimeZone getLocalTimeZone()
{
   // this function is not reentrant, but it's ok, as
   // the worst thing that may happen in MT or multiprocess
   // is double calculation of the cached value.

   // infer timezone by checking gmtime/localtime
   if( s_cached_timezone == tz_local )
   {
      struct tm gmt;
      struct tm loc;
      time_t now;
      time( &now );
      localtime_r( &now, &loc );
      gmtime_r( &now, &gmt );

      // great, let's determine the time shift
      int64 tgmt = mktime( &gmt );
      int64 tloc = mktime( &loc );
      long shift = (long) (tloc - tgmt);
      if ( loc.tm_isdst )
      {
         shift += 3600;
      }

      long hours = shift / 3600;
      long minutes = (shift/60) % 60;


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
}

numeric seconds()
{
   struct timeval time;
   gettimeofday( &time, 0 );
   return time.tv_sec + (time.tv_usec / 1000000.0 );
}

bool absoluteWait( const TimeStamp &ts )
{
   TimeStamp now;

   currentTime( now );
   if ( ts <= now )
      return false;

   TimeStamp diff = ts - now;

   struct timespec tw;
   tw.tv_nsec = diff.m_msec *10000000;
   tw.tv_sec = (time_t) diff.m_day * (24*3600) + diff.m_hour * 3600 +
      diff.m_minute * 60 + diff.m_second;

   if ( nanosleep( &tw, 0 ) == -1 )
      return false;

   return true;
}

bool relativeWait( const TimeStamp &ts )
{
   struct timespec tw;
   tw.tv_nsec = ts.m_msec *10000000;
   tw.tv_sec = (time_t) ts.m_day * (24*3600) + ts.m_hour * 3600 + ts.m_minute * 60 + ts.m_second;

   if ( nanosleep( &tw, 0 ) == -1 )
      return false;

   return true;
}

bool nanoWait( int32 seconds, int32 nanoseconds )
{
   struct timespec tw;
   tw.tv_nsec = nanoseconds;
   tw.tv_sec = (time_t) seconds;

   if ( nanosleep( &tw, 0 ) == -1 )
      return false;

   return true;
}

void timestampFromSystemTime( const SystemTime &sys_time, TimeStamp &ts )
{
   struct tm date;
   const UnixSystemTime *unix_time = static_cast< const UnixSystemTime *>( &sys_time );

   localtime_r( & unix_time->m_time_t, &date );
   ts.m_year = date.tm_year + 1900;
   ts.m_month = date.tm_mon+1;
   ts.m_day = date.tm_mday;
   ts.m_hour = date.tm_hour;
   ts.m_minute = date.tm_min;
   ts.m_second = date.tm_sec;
   // todo: collect day of year and weekday
   ts.m_timezone = tz_local;
}

} // time
} // sys

}


/* end of time_sys_unix.cpp */
