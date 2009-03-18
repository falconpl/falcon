/*
   FALCON - The Falcon Programming Language.
   FILE: sys_time.h

   Time related system service interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun mar 6 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Time related system service interface.
*/

#ifndef flc_sys_time_H
#define flc_sys_time_H

namespace Falcon {


/** Empty SystemTime class.
   This will be overloaded by actual OS system time, and it will
   be used as an opaque object in the whole TimeStamp api; it
   will be decoded by the system-level functions into a
   system independent "TimeStamp" object.
*/
class SystemTime {
};

typedef enum {
      tz_local = 0,
      tz_UTC = 1,
      tz_UTC_E_1 = 2,
      tz_UTC_E_2 = 3,
      tz_UTC_E_3 = 4,
      tz_UTC_E_4 = 5,
      tz_UTC_E_5 = 6,
      tz_UTC_E_6 = 7,
      tz_UTC_E_7 = 8,
      tz_UTC_E_8 = 9,
      tz_UTC_E_9 = 10,
      tz_UTC_E_10 = 11,
      tz_UTC_E_11 = 12,
      tz_UTC_E_12 = 13,
      tz_UTC_W_1 = 14,
      tz_UTC_W_2 = 15,
      tz_UTC_W_3 = 16,
      tz_UTC_W_4 = 17,
      tz_UTC_W_5 = 18,
      tz_UTC_W_6 = 19,
      tz_UTC_W_7 = 20,
      tz_UTC_W_8 = 21,
      tz_UTC_W_9 = 22,
      tz_UTC_W_10 = 23,
      tz_UTC_W_11 = 24,
      tz_UTC_W_12 = 25,
      /** Norfolk (Island) Time	UTC + 11:30 hours */
      tz_NFT = 26,
      /** Australian Central Daylight Time	UTC + 10:30 hours */
      tz_ACDT = 27,
      /** Australian Central Standard Time	UTC + 9:30 hours */
      tz_ACST = 28,
      /** Advanced time of Terre-Neuve	UTC - 2:30 hours */
      tz_HAT = 29,
      /** Newfoundland Standard Time	UTC - 3:30 hours */
      tz_NST = 30,
      /** No zone. Used for date differences */
      tz_NONE = 31
   } TimeZone;

// forward decl
class TimeStamp;

namespace Sys {


namespace Time {

void FALCON_DYN_SYM currentTime( ::Falcon::TimeStamp &ts );

TimeZone FALCON_DYN_SYM getLocalTimeZone();

numeric FALCON_DYN_SYM seconds();
bool FALCON_DYN_SYM absoluteWait( const TimeStamp &ts );
bool FALCON_DYN_SYM relativeWait( const TimeStamp &ts );
bool FALCON_DYN_SYM nanoWait( int32 seconds, int32 nanoseconds );
void FALCON_DYN_SYM timestampFromSystemTime( const SystemTime &st, ::Falcon::TimeStamp &ts );

} // time
} // sys
} // falcon

#endif

/* end of sys_time.h */
