/*
   FALCON - The Falcon Programming Language.
   FILE: TimeStampapi.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun mar 6 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/timestamp.h>
#include <falcon/memory.h>
#include <falcon/time_sys.h>
#include <falcon/autocstring.h>
#include <falcon/item.h>

#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *RFC_2822_days[] = { "Mon","Tue", "Wed","Thu","Fri","Sat","Sun" };

static const char *RFC_2822_months[] = {
   "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug", "Sep","Oct","Nov","Dec" };

namespace Falcon {

inline bool i_isLeapYear( int year )
{
   if ( year % 100 == 0 ) {
      // centuries that can be divided by 400 are leap
      // others are not.
      return ( year % 400 == 0 );
   }

   // all other divisible by 4 years are leap
   return ( year % 4 == 0 );
}

void TimeStamp::copy( const TimeStamp &ts )
{
   m_year = ts.m_year;
   m_month = ts.m_month;
   m_day = ts.m_day;
   m_timezone = ts.m_timezone;
   m_hour = ts.m_hour;
   m_minute = ts.m_minute;
   m_second = ts.m_second;
   m_msec = ts.m_msec;
}

const char *TimeStamp::getRFC2822_ZoneName( TimeZone tz, bool bSemantic, bool bSaving )
{
   if ( tz == tz_local )
      tz = Sys::Time::getLocalTimeZone();

   switch( tz )
   {
      case tz_local: return "+????";
      case tz_NONE: case tz_UTC: if ( bSemantic ) return "GMT"; return "+0000";
      case tz_UTC_E_1: return "+0100";
      case tz_UTC_E_2: return "+0200";
      case tz_UTC_E_3: return "+0300";
      case tz_UTC_E_4: return "+0400";
      case tz_UTC_E_5: return "+0500";
      case tz_UTC_E_6: return "+0600";
      case tz_UTC_E_7: return "+0700";
      case tz_UTC_E_8: return "+0800";
      case tz_UTC_E_9: return "+0900";
      case tz_UTC_E_10: return "+1000";
      case tz_UTC_E_11: return "+1100";
      case tz_UTC_E_12: return "+1200";

      case tz_UTC_W_1: return "-0100";
      case tz_UTC_W_2: return "-0200";
      case tz_UTC_W_3: return "-0300";
      case tz_UTC_W_4: if ( bSemantic ) return "EDT"; return "-0400";
      case tz_UTC_W_5: if ( bSemantic ) return bSaving ? "EST":"CDT"; return "-0500";
      case tz_UTC_W_6: if ( bSemantic ) return bSaving ? "CST":"MDT"; return "-0600";
      case tz_UTC_W_7: if ( bSemantic ) return bSaving ? "MST":"PDT"; return "-0700";
      case tz_UTC_W_8: if ( bSemantic ) return "PST"; return "-0800";
      case tz_UTC_W_9: return "-0900";
      case tz_UTC_W_10: return "-1000";
      case tz_UTC_W_11: return "-1100";
      case tz_UTC_W_12: return "-1200";

      case tz_NFT: return "+1130";
      case tz_ACDT: return "+1030";
      case tz_ACST: return "+0930";
      case tz_HAT: return "-0230";
      case tz_NST: return "-0330";
   }
   return "+????";
}


TimeZone TimeStamp::getRFC2822_Zone( const char *csZoneName )
{
   if( strncmp( "UT", csZoneName, 2 ) == 0 ||
      strncmp( "GMT", csZoneName, 3 ) == 0 ||
      strncmp( "+0000", csZoneName, 5 ) == 0 )
   {
      return tz_UTC;
   }

   if ( csZoneName[0] == '+' )
   {
      int zone = atoi( csZoneName + 1 );
      switch( zone )
      {
         case 100: return tz_UTC_E_1;
         case 200: return tz_UTC_E_2;
         case 300: return tz_UTC_E_3;
         case 400: return tz_UTC_E_4;
         case 500: return tz_UTC_E_5;
         case 600: return tz_UTC_E_6;
         case 700: return tz_UTC_E_7;
         case 800: return tz_UTC_E_8;
         case 900: return tz_UTC_E_9;
         case 1000: return tz_UTC_E_10;
         case 1100: return tz_UTC_E_11;
         case 1200: return tz_UTC_E_12;

         case 1130: return tz_NFT;
         case 1030: return tz_ACDT;
         case 930: return tz_ACST;
      }
      return tz_NONE;
   }
   else if ( csZoneName[0] == '-' )
   {
      int zone = atoi( csZoneName + 1 );
      switch( zone )
      {
         case 100: return tz_UTC_W_1;
         case 200: return tz_UTC_W_2;
         case 300: return tz_UTC_W_3;
         case 400: return tz_UTC_W_4;
         case 500: return tz_UTC_W_5;
         case 600: return tz_UTC_W_6;
         case 700: return tz_UTC_W_7;
         case 800: return tz_UTC_W_8;
         case 900: return tz_UTC_W_9;
         case 1000: return tz_UTC_W_10;
         case 1100: return tz_UTC_W_11;
         case 1200: return tz_UTC_W_12;
         case 230: return tz_HAT;
         case 330: return tz_NST;
      }
      return tz_NONE;
   }

   if( strncmp( "EDT", csZoneName, 3 ) == 0 )
   {
      return tz_UTC_W_4;
   }

   if( strncmp( "EST", csZoneName, 3 ) == 0 ||
      strncmp( "CDT", csZoneName, 3 ) == 0 )
   {
      return tz_UTC_W_5;
   }

   if( strncmp( "CST", csZoneName, 3 ) == 0 ||
      strncmp( "MDT", csZoneName, 3 ) == 0 )
   {
      return tz_UTC_W_6;
   }

   if( strncmp( "MST", csZoneName, 3 ) == 0 ||
      strncmp( "PDT", csZoneName, 3 ) == 0 )
   {
      return tz_UTC_W_7;
   }

   if( strncmp( "PST", csZoneName, 3 ) == 0 )
   {
      return tz_UTC_W_8;
   }

   // failure
   return tz_NONE;
}

const char *TimeStamp::getRFC2822_WeekDayName( int16 wd )
{
   if ( wd >=0 && wd < 7 )
   {
      return RFC_2822_days[wd];
   }
   return "???";
}

const char *TimeStamp::getRFC2822_MonthName( int16 month )
{
   if ( month >= 1 && month <= 12 )
   {
      return RFC_2822_months[ month - 1 ];
   }
   return "???";
}


int16 TimeStamp::getRFC2822_WeekDay( const char *name )
{
   for ( uint32 i = 0; i < sizeof( RFC_2822_days ) / sizeof( char *); i ++ )
   {
      if ( strncmp( RFC_2822_days[i], name, 3) == 0 )
         return i;
   }
   return -1;
}

int16 TimeStamp::getRFC2822_Month( const char *name )
{
   for ( uint32 i = 0; i < sizeof( RFC_2822_months ) / sizeof( char *); i ++ )
   {
      if ( strncmp( RFC_2822_months[i], name, 3) == 0 )
         return i+1;
   }
   return -1;
}

TimeStamp &TimeStamp::operator = ( const TimeStamp &ts )
{
   copy(ts);
   return *this;
}


bool TimeStamp::toRFC2822( String &target, bool bSemantic, bool bDst ) const
{
   if ( ! isValid() )
   {
      target = "?";
      return false;
   }

   target.append( getRFC2822_WeekDayName( dayOfWeek() ) );
   target.append( ',' );
   target.append( ' ' );

   target.writeNumber( (int64) m_day, "%02d" );

   target.append( ' ' );
   target.append( getRFC2822_MonthName( m_month ) );
   target.append( ' ' );
   if ( m_year < 0 )
      target.append( "0000" );
   else {
      target.writeNumber( (int64) m_year, "%04d" );
   }

   target.append( ' ' );
   target.writeNumber( (int64) m_hour, "%02d" );
   target.append( ':' );
   target.writeNumber( (int64) m_minute, "%02d" );
   target.append( ':' );
   target.writeNumber( (int64) m_second, "%02d" );

   target.append( ' ' );
   TimeZone tz = m_timezone;

   if  ( tz == tz_local )
   {
      tz = Sys::Time::getLocalTimeZone();
   }

   target.append( getRFC2822_ZoneName( tz, bSemantic, bDst ) );

   return true;
}


bool TimeStamp::fromRFC2822( TimeStamp &target, const String &source )
{
   AutoCString cstr( source );
   return fromRFC2822( target, cstr.c_str() );
}

bool TimeStamp::fromRFC2822( TimeStamp &target, const char *source )
{
   const char *pos = source;

   // Find the comma
   while ( *pos != 0 && *pos != ',' ) pos++;
   if ( *pos == 0 || (pos-source)!= 3 )
      return false;

   // is this a valid day?
   if( getRFC2822_WeekDay( source ) < 0 )
      return false;

   pos++;
   if ( *pos == 0 )
      return false;
   pos++;
   const char *mon = pos;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2)
      return false;
   target.m_day = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 3)
      return false;
   target.m_month = getRFC2822_Month( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 4)
      return false;
   target.m_year = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ':' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   target.m_hour = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ':' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   target.m_minute = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   target.m_second = atoi( pos );

   mon++;
   target.m_timezone = getRFC2822_Zone( mon );
   if( target.m_timezone == tz_NONE )
      return false;

   target.m_msec = 0;

   return target.isValid();
}

bool TimeStamp::isValid() const
{
   if ( m_msec < 0 || m_msec >= 1000 )
      return false;
   if ( m_second < 0 || m_second >= 60 )
      return false;
   if ( m_minute < 0 || m_minute >= 60 )
      return false;
   if ( m_hour < 0 || m_hour >= 24 )
      return false;
   if ( m_month < 1 || m_month > 12 )
      return false;
   if ( m_day < 1 || m_day > 31 )
      return false;

   // exclude months with 31 days
   if ( m_month == 1 || m_month == 3 || m_month == 5 || m_month == 7 || m_month == 8 ||
        m_month == 10|| m_month == 12 )
      return true;

   // now exclude the months with 30 days
   if ( m_month != 2 ) {
      if ( m_day != 31 )
         return true;
      return false;  // 31 days in a 30 days long month
   }

   // february.
   if ( m_day == 30 )
      return false;
   if ( m_day < 29 )
      return true;

   // we have to calculate the leap year
   if ( isLeapYear() )
      return true;

   return false;
}

bool TimeStamp::isLeapYear() const
{
   if ( m_year % 100 == 0 ) {
      // centuries that can be divided by 400 are leap
      // others are not.
      return ( m_year % 400 == 0 );
   }

   // all other divisible by 4 years are leap
   return ( m_year % 4 == 0 );
}


int16 TimeStamp::dayOfYear() const
{
   if ( ! isValid() )
      return 0;

   int days = 0;
   // add all the previous days
   switch( m_month ) {
      case 12: days += 30;
      case 11: days += 31;
      case 10: days += 30;
      case 9: days += 31;
      case 8: days += 31;
      case 7: days += 30;
      case 6: days += 31;
      case 5: days += 30;
      case 4: days += 31;
      case 3: days += 28; if ( isLeapYear() ) days ++;
      case 2: days += 31;
   }
   days += m_day;

   return days;
}

int16 TimeStamp::getDaysOfMonth( int16 month ) const
{
   if ( month == -1 ) month = m_month;

   switch( month ) {
      case 1: return 31;
      case 2: return isLeapYear()? 29 : 28;
      case 3: return 31;
      case 4: return 30;
      case 5: return 31;
      case 6: return 30;
      case 7: return 31;
      case 8: return 31;
      case 9: return 30;
      case 10: return 31;
      case 11: return 30;
      case 12: return 31;
   }
   return 0;
}

/** week starting on monday, 0 based. */
int16 TimeStamp::dayOfWeek() const
{
   // consider 1700 the epoch
   if ( m_year < 1700 )
      return -1;

   // compute days since epoch.
   int32 nyears = m_year - 1700;
   int32 nday = nyears * 365;

   // add leap years (up to the previous year. This year is  computed in dayOfYear()
   if( m_year > 1700 )
      nday += ((nyears-1) / 4) - ((nyears-1) / 100) + ((nyears-1) / 400);

   // add day of the year.
   nday += dayOfYear();

   // add epoch (1/1/1700 was a Friday)
   nday += 4;

   nday %= 7;
   return nday;
}


int64 TimeStamp::toLongFormat() const
{
   if ( ! isValid() )
      return -1;

   int64 res = 0;

   res |= m_year;
   // 4 bits for months
   res <<= 4;
   res |= m_month;

   // 5 bits for days
   res <<= 5;
   res |= m_day;

   // 5 bits for hours
   res <<= 5;
   res |= m_hour;

   // 6 bits for minutes
   res <<= 6;
   res |= m_minute;
   // 6 bits for seconds
   res <<= 6;
   res |= m_second;
   //10 bits for msecs
   res <<= 10;
   res |= m_msec;
   // 5 bits for tz.
   res <<= 5;
   res |= (int) m_timezone;

   return res;
}

void TimeStamp::fromLongFormat( int64 lf )
{
   m_timezone = (TimeZone) (0x1f & lf);
   lf >>= 5;
   m_msec = (int16) ( 0x3ff & lf );
   lf >>= 10;
   m_second = (int16) ( 0x3f & lf );
   lf >>= 6;
   m_minute = (int16) ( 0x3f & lf );
   lf >>= 6;
   m_hour = (int16) ( 0x1f & lf );
   lf >>= 5;
   m_day = (int16) ( 0x1f & lf );
   lf >>= 5;
   m_month = (int16) ( 0xf & lf );
   lf >>= 4;
   m_year = (int16) lf;
}


void TimeStamp::add( const TimeStamp &ts )
{
   m_day += ts.m_day;
   m_hour += ts.m_hour;
   m_minute += ts.m_minute;
   m_second += ts.m_second;
   m_msec += ts.m_msec;

   if ( m_timezone != ts.m_timezone && m_timezone != tz_NONE && ts.m_timezone != tz_NONE )
   {
      int16 hours=0, mins=0, ts_hours=0, ts_mins=0;
      ts.getTZDisplacement( ts_hours, ts_mins );
      getTZDisplacement( hours, mins );
      m_hour += hours - ts_hours;
      m_minute += hours - ts_mins;
   }

   rollOver();
   if ( m_timezone == tz_NONE )
      m_timezone = ts.m_timezone;
}

void TimeStamp::add( int32 days, int32 hours, int32 mins, int32 secs, int32 msecs )
{
   m_day = days + dayOfYear();
   m_hour += hours;
   m_minute += mins;
   m_second += secs;
   m_msec += msecs;

   rollOver();
}

inline void cplxSub( int &sub1, int sub2, int unit, int &change )
{
   sub1 -= sub2;
   while( sub1 < 0 )
   {
      change--;
      sub1 += unit;
   }
}

void TimeStamp::distance( const TimeStamp &ts )
{
   int days = 0;

   // first decide which date is bigger.
   const TimeStamp *startDate, *endDate;
   int comparation = this->compare( ts );
   if (comparation == 0 ) {
      // the same date, means no distance.
      m_msec = m_second = m_minute = m_hour = m_day = m_month = m_year = 0;
      return;
   }

   if ( comparation > 0 ) {
      startDate = &ts;
      endDate = this;
   }
   else {
      startDate = this;
      endDate = &ts;
   }

   // If year is different:
   if( startDate->m_year != endDate->m_year )
   {
      // calculate the number of days in the in-between years
      for ( int baseYear = startDate->m_year + 1; baseYear < endDate->m_year; baseYear++ )
         days += i_isLeapYear( baseYear ) ? 366 : 365;

      // calculate the number of days from start day to the end of the year.
      int doy = ( startDate->isLeapYear() ? 366 : 365 ) - startDate->dayOfYear();
      days += doy;

      // and add the days in the year of the target date
      days += endDate->dayOfYear();
   }
   else {
      days += endDate->dayOfYear() - startDate->dayOfYear();
   }

   m_year = 0;
   m_month = 0;

   //int tday = days;
   int thour = endDate->m_hour;
   int tminute = endDate->m_minute;
   int tsecond = endDate->m_second;
   int tmsec = endDate->m_msec;

   if ( m_timezone != ts.m_timezone && m_timezone != tz_NONE && ts.m_timezone != tz_NONE )
   {
      int16 hours=0, mins=0, ts_hours=0, ts_mins=0;
      ts.getTZDisplacement( ts_hours, ts_mins );
      getTZDisplacement( hours, mins );
      // if ts bigger (positive distance) we must add the difference between TS timezone and us
      if ( comparation < 0 )
      {
         thour += ts_hours - hours;
         tminute += ts_mins - mins;
      }
      else {
         // else we got to subtract it
         thour += ts_hours - hours;
         tminute += ts_mins - mins;
      }
   }

   cplxSub( tmsec, startDate->m_msec, 1000, tsecond );
   cplxSub( tsecond, startDate->m_second, 60, tminute );
   cplxSub( tminute, startDate->m_minute, 60, thour );
   cplxSub( thour, startDate->m_hour, 24, days );

   m_day = days;
   m_hour = thour;
   m_minute = tminute;
   m_second = tsecond;
   m_msec = tmsec;

   if( comparation > 0 )
   {
      // the negative sign goes on the first non-zero unit
      if ( m_day != 0 )
         m_day = -m_day;
      else if ( m_hour != 0 )
         m_hour = -m_hour;
      else if ( m_minute != 0 )
         m_minute = -m_minute;
      else if ( m_second != 0 )
         m_second = -m_second;
      else
         m_msec = -m_msec;
   }

   m_timezone = tz_NONE;
}

void TimeStamp::rollOver( bool onlyDays )
{
   // do rollovers
   int32 adjust = 0;
   if( m_msec < 0 ) {
      adjust = - ( (-m_msec) / 1000 + 1 );
      m_msec = 1000 - ((-m_msec)%1000);
   }

   if ( m_msec >= 1000 ) {
      adjust += m_msec / 1000;
      m_msec = m_msec % 1000;
   }

   m_second += adjust;
   adjust = 0;
   if( m_second < 0 ) {
     adjust = - ( (-m_second) / 60 + 1 );
     m_second = 60 - ((-m_second)%60);
   }

   if (m_second >= 60 ) {
     adjust += m_second / 60;
     m_second = m_second % 60;
   }

   m_minute += adjust;
   adjust = 0;
   if( m_minute < 0 ) {
      adjust = - ( (-m_minute) / 60 + 1 );
      m_minute = 60 - ((-m_minute)%60);
   }

   if ( m_minute >= 60 ) {
      adjust += m_minute / 60;
      m_minute = m_minute % 60;
   }

   m_hour += adjust;
   adjust = 0;
   if( m_hour < 0 ) {
      adjust = - ( (-m_hour) / 24 + 1 );
      m_hour = 24 - ((-m_hour)%24);
   }

   if ( m_hour >= 24 ) {
      adjust += m_hour / 24;
      m_hour = m_hour % 24;
   }

   m_day += adjust;
   if ( onlyDays ) {
      return;
   }

   if ( m_day > 0 )
   {
      int16 mdays;
      while( m_day > (mdays = getDaysOfMonth( m_month )) )
      {
         m_day -= mdays;

         if( m_month == 12 )
         {
            m_month = 1;
            m_year++;
         }
         else
            m_month++;
      }
   }
   else {
      while( m_day < 1 )
      {
         if ( m_month == 1 )
         {
            m_month = 12;
            m_year--;
         }
         else
            m_month --;
         m_day += getDaysOfMonth( m_month );
      }
   }
}

int32 TimeStamp::compare( const TimeStamp &ts ) const
{
   if ( m_year < ts.m_year  ) return -1;
   if ( m_year > ts.m_year  ) return 1;

   if ( m_month < ts.m_month  ) return -1;
   if ( m_month > ts.m_month  ) return 1;

   if ( m_day < ts.m_day  ) return -1;
   if ( m_day > ts.m_day  ) return 1;

   if ( ts.m_timezone == m_timezone || ts.m_timezone == tz_NONE || m_timezone == tz_NONE )
   {
      if ( m_hour < ts.m_hour  ) return -1;
      if ( m_hour > ts.m_hour  ) return 1;

      if ( m_day < ts.m_day  ) return -1;
      if ( m_day > ts.m_day  ) return 1;
   }
   else {
      int16 hdisp=0, mdisp=0;
      int16 ts_hdisp=0, ts_mdisp=0;\

      getTZDisplacement( hdisp, mdisp );
      ts.getTZDisplacement( ts_hdisp, ts_mdisp );

      if ( m_hour + hdisp < ts.m_hour + ts_hdisp ) return -1;
      if ( m_hour + hdisp > ts.m_hour + ts_hdisp ) return 1;

      if ( m_day + mdisp < ts.m_day + ts_mdisp ) return -1;
      if ( m_day + mdisp > ts.m_day + ts_mdisp ) return 1;
   }

   if ( m_minute < ts.m_minute  ) return -1;
   if ( m_minute > ts.m_minute  ) return 1;

   if ( m_second < ts.m_second  ) return -1;
   if ( m_second > ts.m_second  ) return 1;

   if ( m_msec < ts.m_msec  ) return -1;
   if ( m_msec > ts.m_msec  ) return 1;

   return 0;
}

void TimeStamp::getTZDisplacement( int16 &hours, int16 &minutes ) const
{
   TimeZone tz = m_timezone;
   getTZDisplacement( tz, hours, minutes );
}


void TimeStamp::getTZDisplacement( TimeZone tz, int16 &hours, int16 &minutes )
{
   if  ( tz == tz_local )
   {
      tz = Sys::Time::getLocalTimeZone();
   }

   switch( tz )
   {
      case tz_local: case tz_NONE: case tz_UTC: hours = 0; minutes = 0; break;
      case tz_UTC_E_1: hours = 1; minutes = 0; break;
      case tz_UTC_E_2: hours = 2; minutes = 0; break;
      case tz_UTC_E_3: hours = 3; minutes = 0; break;
      case tz_UTC_E_4: hours = 4; minutes = 0; break;
      case tz_UTC_E_5: hours = 5; minutes = 0; break;
      case tz_UTC_E_6: hours = 6; minutes = 0; break;
      case tz_UTC_E_7: hours = 7; minutes = 0; break;
      case tz_UTC_E_8: hours = 8; minutes = 0; break;
      case tz_UTC_E_9: hours = 9; minutes = 0; break;
      case tz_UTC_E_10: hours = 10; minutes = 0; break;
      case tz_UTC_E_11: hours = 11; minutes = 0; break;
      case tz_UTC_E_12: hours = 12; minutes = 0; break;

      case tz_UTC_W_1: hours = -1; minutes = 0; break;
      case tz_UTC_W_2: hours = -2; minutes = 0; break;
      case tz_UTC_W_3: hours = -3; minutes = 0; break;
      case tz_UTC_W_4: hours = -4; minutes = 0; break;
      case tz_UTC_W_5: hours = -5; minutes = 0; break;
      case tz_UTC_W_6: hours = -6; minutes = 0; break;
      case tz_UTC_W_7: hours = -7; minutes = 0; break;
      case tz_UTC_W_8: hours = -8; minutes = 0; break;
      case tz_UTC_W_9: hours = -9; minutes = 0; break;
      case tz_UTC_W_10: hours = -10; minutes = 0; break;
      case tz_UTC_W_11: hours = -11; minutes = 0; break;
      case tz_UTC_W_12: hours = -12; minutes = 0; break;

      case tz_NFT: hours = 11; minutes = 30; break;
      case tz_ACDT: hours = 10; minutes = 30; break;
      case tz_ACST: hours = 9; minutes = 30; break;
      case tz_HAT: hours = -2; minutes = -30; break;
      case tz_NST: hours = -3; minutes = -30; break;
   }
}

void TimeStamp::toString( String &target ) const
{
   // for now a fast thing
   uint32 allocated = 23 > FALCON_STRING_ALLOCATION_BLOCK ? 24 : FALCON_STRING_ALLOCATION_BLOCK;
   char *storage = (char *) memAlloc( allocated );
   sprintf( (char *)storage, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
      m_year, m_month, m_day, m_hour, m_minute, m_second, m_msec );
   target.adopt( storage, 23, allocated );
}

bool TimeStamp::toString( String &target, const String &fmt ) const
{
   AutoCString cfmt( fmt );
   struct tm theTime;

   theTime.tm_sec = m_second;
   theTime.tm_min = m_minute;
   theTime.tm_hour = m_hour;
   theTime.tm_mday = m_day;
   theTime.tm_mon = m_month-1;
   theTime.tm_year = m_year - 1900;

   char timeTgt[512];
   if( strftime( timeTgt, 512, cfmt.c_str(), &theTime) != 0 )
   {
      target.bufferize( timeTgt );

      uint32 pos = target.find( "%i" );
      if( pos !=  String::npos )
      {
         String rfc;
         toRFC2822( rfc );
         while( pos != String::npos )
         {
            target.change( pos, pos + 2, rfc );
            pos = target.find( "%i", pos + 2 );
         }
      }

      pos = target.find( "%q" );
      if( pos !=  String::npos )
      {
         String msecs;
         msecs.writeNumber( (int64) m_msec );
         while( pos != String::npos )
         {
            target.change( pos, pos + 2, msecs );
            pos = target.find( "%q", pos + 2 );
         }
      }

      pos = target.find( "%Q" );
      if( pos !=  String::npos )
      {
         String msecs;
         if( m_msec < 10 )
            msecs = "00";
         else if ( m_msec < 100 )
            msecs = "0";

         msecs.writeNumber( (int64) m_msec );

         while( pos != String::npos )
         {
            target.change( pos, pos + 2, msecs );
            pos = target.find( "%Q", pos + 2 );
         }
      }

      return true;
   }

   return false;
}

void TimeStamp::currentTime()
{
   Sys::Time::currentTime( *this );
   if  ( m_timezone == tz_local )
   {
      m_timezone = Sys::Time::getLocalTimeZone();
   }
}

void TimeStamp::changeTimezone( TimeZone tz )
{
   if  ( m_timezone == tz_local )
   {
      m_timezone = Sys::Time::getLocalTimeZone();

      if (tz == tz_local ) // no shift...
         return;
   }

   if ( tz == tz_local )
      tz = Sys::Time::getLocalTimeZone();

   // no shift
   if ( tz == m_timezone )
      return;

   // get the relative total shift.
   int16 currentHour=0, currentMin=0, newHour=0, newMin=0;
   getTZDisplacement( tz, newHour, newMin );
   getTZDisplacement( m_timezone, currentHour, currentMin );

   m_hour -= currentHour;
   m_hour += newHour;
   m_minute -= currentMin;
   m_minute += newMin;
   rollOver( false );
   m_timezone = tz;
}

TimeStamp *TimeStamp::clone() const
{
   return new TimeStamp( *this );
}

}

/* end of TimeStampapi.cpp */
