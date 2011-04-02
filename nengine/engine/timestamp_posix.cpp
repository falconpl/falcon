/*
   FALCON - The Falcon Programming Language.
   FILE: sys_posix.cpp

   System specific (unix) support for VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <falcon/timestamp.h>
#include <falcon/interruptederror.h>

namespace Falcon
{

static TimeStamp::TimeZone s_cached_timezone = TimeStamp::tz_local; // which is also 0

void TimeStamp::setCurrent(bool bLocal)
{
   struct timeval current;
   struct tm date;
   time_t t;

   gettimeofday( &current, 0 );
   time( &t );
   
   m_msec = current.tv_usec / 1000;
   if( bLocal )
   {
     localtime_r( &t, &date );
     m_timezone = tz_local;
   }
   else
   {
     gmtime_r( &t, &date );
     m_timezone = tz_UTC;

   }
   m_year = date.tm_year + 1900;
   m_month = date.tm_mon+1;
   m_day = date.tm_mday;
   m_hour = date.tm_hour;
   m_minute = date.tm_min;
   m_second = date.tm_sec;



}


TimeStamp::TimeZone TimeStamp::getLocalTimeZone()
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


bool TimeStamp::absoluteWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   TimeStamp now;
   now.setCurrent();

   if ( ts <= now )
      return false;

   TimeStamp diff = ts - now;

   struct timeval tv;
   fd_set set;

   int* pipe_fds = (int*) intr->sysData();
   FD_SET( pipe_fds[0], &set );

   tv.tv_usec = diff.m_msec *1000;
   tv.tv_sec = (time_t) diff.m_day * (24*3600) + diff.m_hour * 3600 +
      diff.m_minute * 60 + diff.m_second;

   if( select( pipe_fds[0] + 1, &set, 0, 0, &tv ) > 0 )
   {
      intr->reset();
      throw new InterruptedError( ErrorParam( e_interrupted, __LINE__, __FILE__ ) );
   }

   return true;
}


bool TimeStamp::absoluteWait( const TimeStamp &ts )
{
   TimeStamp now;
   now.setCurrent();

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


bool TimeStamp::relativeWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   struct timeval tv;
   fd_set set;

   int* pipe_fds = (int*) intr->sysData();
   FD_SET( pipe_fds[0], &set );

   tv.tv_usec = ts.m_msec *1000;
   tv.tv_sec = (time_t) ts.m_day * (24*3600) + ts.m_hour * 3600 +
      ts.m_minute * 60 + ts.m_second;

   if( select( pipe_fds[0] + 1, &set, 0, 0, &tv ) > 0 )
   {
      intr->reset();
      throw new InterruptedError( ErrorParam( e_interrupted, __LINE__, __FILE__ ) );
   }

   return true;
}


bool TimeStamp::relativeWait( const TimeStamp &ts )
{   
   struct timespec tw;
   tw.tv_nsec = ts.m_msec *10000000;
   tw.tv_sec = (time_t) ts.m_day * (24*3600) + ts.m_hour * 3600 + ts.m_minute * 60 + ts.m_second;

   if ( nanosleep( &tw, 0 ) == -1 )
      return false;

   return true;
}


void TimeStamp::fromSystemTime( void* sys_ts )
{
   struct tm date;
   localtime_r( (time_t*) sys_ts, &date );

   m_year = date.tm_year + 1900;
   m_month = date.tm_mon+1;
   m_day = date.tm_mday;
   m_hour = date.tm_hour;
   m_minute = date.tm_min;
   m_second = date.tm_sec;

   // todo: collect day of year and weekday
   m_timezone = tz_local;
}

}

/* end of timestamp_posix.cpp */
