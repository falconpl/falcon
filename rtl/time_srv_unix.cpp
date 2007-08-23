/*
   FALCON - The Falcon Programming Language.
   FILE: time_srv_unix.cpp
   $Id: time_srv_unix.cpp,v 1.1.1.1 2006/10/08 15:05:06 gian Exp $

   Unix-specific time related services (only timestamp needed).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006
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
   Unix-specific time related services (only timestamp needed).
*/

#include "time_sys_unix.h"

namespace Falcon {

void TimeStamp::fromSystemTime( const SystemTime &sys_time )
{
   struct tm date;
   const UnixSystemTime *unix_time = static_cast< const UnixSystemTime *>( &sys_time );

   localtime_r( & unix_time->m_time_t, &date );
   m_year = date.tm_year + 1900;
   m_month = date.tm_mon+1;
   m_day = date.tm_mday;
   m_hour = date.tm_hour;
   m_minute = date.tm_min;
   m_second = date.tm_sec;
   // todo: collect day of year and weekday
   m_timezone = ::Falcon::TimeStamp::tz_local;
}

}


/* end of time_srv_unix.cpp */
