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
#include <falcon/interruptederror.h>
#include <windows.h>

namespace Falcon
{

static TimeStamp::TimeZone s_cached_timezone = TimeStamp::tz_local; // which is also 0

void TimeStamp::setCurrent(bool bLocal)
{
}


TimeStamp::TimeZone TimeStamp::getLocalTimeZone()
{
   return s_cached_timezone;
}


bool TimeStamp::absoluteWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   return true;
}


bool TimeStamp::absoluteWait( const TimeStamp &ts )
{
	return true;
}


bool TimeStamp::relativeWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr )
{
   return true;
}


bool TimeStamp::relativeWait( const TimeStamp &ts )
{   
   return true;
}


void TimeStamp::fromSystemTime( void* sys_ts )
{
}

}

/* end of timestamp_win.cpp */
