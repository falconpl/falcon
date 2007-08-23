/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: time_srv_win.cpp
   $Id: time_srv_win.cpp,v 1.1.1.1 2006/10/08 15:05:00 gian Exp $

   Win-specific time related services (only timestamp needed).
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
   Win-specific time related services (only timestamp needed).
*/

#include "time_sys_win.h"

namespace Falcon {

void TimeStamp::fromSystemTime( const SystemTime &sys_time )
{
   const WinSystemTime *win_time = static_cast< const WinSystemTime *>( &sys_time );

   m_year = win_time->m_time.wYear;
   m_month = win_time->m_time.wMonth;
   m_day = win_time->m_time.wDay;
   m_hour = win_time->m_time.wHour;
   m_minute = win_time->m_time.wMinute;
   m_second = win_time->m_time.wSecond;
   m_msec = win_time->m_time.wMilliseconds;
    
   // todo: collect day of year and weekday
   m_timezone = ::Falcon::TimeStamp::tz_local;
}

}


/* end of time_srv_win.cpp */
