/*
   FALCON - The Falcon Programming Language.
   FILE: timestamp_win.cpp

   System specific (Windows) support for VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/timestamp.h>
#include <windows.h>

namespace Falcon
{

static TimeStamp::TimeZone s_cached_timezone = TimeStamp::tz_local; // which is also 0

void TimeStamp::setCurrent(bool bLocal)
{
   (void) bLocal;
}


TimeStamp::TimeZone TimeStamp::getLocalTimeZone()
{
   return s_cached_timezone;
}


bool TimeStamp::absoluteWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   (void) ts; (void) intr;
   return true;
}


bool TimeStamp::absoluteWait( const TimeStamp &ts )
{
   (void) ts;
	return true;
}


bool TimeStamp::relativeWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   (void) ts; (void) intr;
   return true;
}


bool TimeStamp::relativeWait( const TimeStamp &ts )
{
   (void) ts;
   return true;
}


void TimeStamp::fromSystemTime( void* sys_ts )
{

   SYSTEMTIME* st = (SYSTEMTIME*) sys_ts;

   m_year = st->wYear;
   m_month = st->wMonth;
   m_day = st->wDay;
   m_hour = st->wHour;
   m_minute = st->wMinute;
   m_second = st->wSecond;
   m_msec = st->wMilliseconds;

   // todo: collect day of year and weekday
   m_timezone = tz_local;

}

}

/* end of timestamp_win.cpp */
