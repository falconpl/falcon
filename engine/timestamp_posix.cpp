/*
   FALCON - The Falcon Programming Language.
   FILE: timestamp_posix.cpp

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
#include <falcon/stderrors.h>

namespace Falcon
{

static TimeStamp::TimeZone s_cached_timezone = TimeStamp::tz_local; // which is also 0

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

         case -1: s_cached_timezone = tz_UTC_W_1; break;
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
            break;
      }
   }

   return s_cached_timezone;
}

bool TimeStamp::getLocalDST()
{
   struct timeval tv;
   struct timezone tz;
   gettimeofday(&tv, &tz);
   return tz.tz_dsttime != 0;
}
}

/* end of timestamp_posix.cpp */
