/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_unix.cpp
   $Id: time_sys_unix.cpp,v 1.1 2007/06/21 21:54:26 jonnymind Exp $

   Unix system implementation for time related service
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
   Unix system implementation for time related service
*/


#include <sys/time.h>
#include <falcon/timestamp.h>
#include <falcon/time_sys_unix.h>

namespace Falcon {

namespace Sys {
namespace Time {

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
   return tz_local;
}

bool sleep( numeric seconds )
{
   struct timespec tw;
   tw.tv_nsec = ((long) (seconds * 1000.0) % 1000 ) * 1000000;
   tw.tv_sec = (time_t) seconds;
   if( nanosleep( &tw, 0 ) == -1 )
      return false;
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
