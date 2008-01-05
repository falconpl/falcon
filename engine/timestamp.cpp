/*
   FALCON - The Falcon Programming Language.
   FILE: TimeStampapi.cpp
   $Id: timestamp.cpp,v 1.7 2007/08/11 12:15:59 jonnymind Exp $

   Short description
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
   Short description
*/

#include <falcon/timestamp.h>
#include <falcon/memory.h>
#include <falcon/time_sys.h>
#include <falcon/string.h>
#include <falcon/item.h>

#include <stdio.h>
#include <time.h>

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

TimeStamp &TimeStamp::operator = ( const TimeStamp &ts )
{
   copy(ts);
   return *this;
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
   if ( m_year < 1700 )
      return -1;

   int32 nyears = m_year - 1700;

   int32 nday = m_year * 365;
   // add leap years.
   // Starting from 1700, we have 3 non-leap centuries in a row.
   nday += (nyears % 4) - (nyears % 100) + (nyears % 400);
   // add day of the year.
   nday += dayOfYear();
   // add epoch (1st january 1700 was a monday )
   nday += 1;

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

numeric TimeStamp::asSeconds() const
{
   return 0;
}

void TimeStamp::fromSeconds( numeric seconds )
{
}

void TimeStamp::add( const TimeStamp &ts )
{
   m_day = ts.m_day + dayOfYear();
   m_hour += ts.m_hour;
   m_minute += ts.m_minute;
   m_second += ts.m_second;
   m_msec += ts.m_msec;

   if ( m_timezone != ts.m_timezone && m_timezone != tz_NONE && ts.m_timezone != tz_NONE )
   {
      int16 hours, mins, ts_hours, ts_mins;
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
   m_day = comparation < 0 ? days : -days;

   m_hour = endDate->m_hour - startDate->m_hour;
   m_minute = endDate->m_minute - startDate->m_minute;
   m_second = endDate->m_second - startDate->m_second;
   m_msec = endDate->m_msec - startDate->m_msec;

   if ( m_timezone != ts.m_timezone && m_timezone != tz_NONE && ts.m_timezone != tz_NONE )
   {
      int16 hours, mins, ts_hours, ts_mins;
      ts.getTZDisplacement( ts_hours, ts_mins );
      getTZDisplacement( hours, mins );
      // if ts bigger (positive distance) we must add the difference between TS timezone and us
      if ( comparation < 0 )
      {
         m_hour += ts_hours - hours;
         m_minute += ts_mins - mins;
      }
      else {
         // else we got to subtract it
         m_hour -= ts_hours - hours;
         m_minute -= ts_mins - mins;
      }
   }

   rollOver( true );
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

   if ( onlyDays ) {
      // if the day is negative, and we want to know about days,
      // a minus sign here means "less negative date"
      m_day = m_day < 0 ? m_day - adjust: m_day + adjust;
      return;
   }

   m_day += adjust;
   adjust = m_day;

   if ( adjust <= 0 ) {
      m_year --;
      while ( adjust < -366 ) {
         adjust += 365;
         if ( i_isLeapYear( m_year ) )
            adjust++;
         m_year--;
      }

      if ( adjust == -365 && ! i_isLeapYear( m_year ) ) {
         adjust = 0;
         m_year--;
      }

      m_month = 12;
      int16 mdays = getDaysOfMonth( m_month );
      adjust += mdays;
      while( adjust <= 0 ) {
         m_month --;
         if ( m_month == 0 ) {
            m_month = 12;
            m_year--;
         }
         mdays = getDaysOfMonth( m_month );
         adjust += mdays;
      }
   }
   else {
      while ( adjust > 366 ) {
         adjust -= 365;
         if ( i_isLeapYear( m_year ) )
            adjust--;
         m_year++;
      }

      if ( adjust == 365 && ! i_isLeapYear( m_year ) ) {
         adjust = 1;
         m_year ++;
      }

      m_month = 1;
      int16 mdays = getDaysOfMonth( m_month );
      while( adjust > mdays ) {
         m_month ++;
         adjust -= mdays;
         mdays = getDaysOfMonth( m_month );
         if ( m_month > 12 ) {
            m_month = 1;
            m_year++;
         }
      }
   }
   m_day = adjust;
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
      int16 hdisp, mdisp;
      int16 ts_hdisp, ts_mdisp;

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

   if  ( tz == tz_local )
   {
      tz = Sys::Time::getLocalTimeZone();
   }

   switch( tz )
   {
      case tz_NONE: case tz_UTC: hours = 0; minutes = 0; break;
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
   char fmtBuf[256];

   if( ! fmt.toCString( fmtBuf, 256 )  )
   {
      return false;
   }

   struct tm theTime;

   theTime.tm_sec = m_second;
   theTime.tm_min = m_minute;
   theTime.tm_hour = m_hour;
   theTime.tm_mday = m_day;
   theTime.tm_mon = m_month;
   theTime.tm_year = m_year - 1900;

   char timeTgt[512];
   if( strftime( timeTgt, 512, fmtBuf, &theTime) != 0 )
   {
      target.bufferize( timeTgt );

      uint32 pos = target.find( "%q" );
      if( pos !=  String::npos )
      {
         String msecs;
         msecs.writeNumber( (int64) m_msec );
         while( pos != String::npos )
         {
            target.change( pos, pos + 2, &msecs );
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
            target.change( pos, pos + 2, &msecs );
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
}


bool TimeStamp::isReflective()
{
   return true;
}

void TimeStamp::getProperty( const String &propName, Item &prop )
{
   if( propName == "year" )
   {
      prop = (int64) m_year;
   }
   else if( propName == "month" ) {
      prop = (int64) m_month;
   }
   else if( propName == "day" ) {
      prop = (int64) m_day;
   }
   else if( propName == "hour" ) {
      prop = (int64) m_hour;
   }
   else if( propName == "minute" ) {
      prop = (int64) m_minute;
   }
   else if( propName == "second" ) {
      prop = (int64) m_second;
   }
   else if( propName == "msec" ) {
      prop = (int64) m_msec;
   }
}

void TimeStamp::setProperty( const String &propName, Item &prop )
{
   if( propName == "year" )
   {
      m_year = (uint16) prop.forceInteger();
   }
   else if( propName == "month" ) {
      m_month = (uint16) prop.forceInteger();
   }
   else if( propName == "day" ) {
      m_day = (uint16) prop.forceInteger();
   }
   else if( propName == "hour" ) {
      m_hour = (uint16) prop.forceInteger();
   }
   else if( propName == "minute" ) {
      m_minute = (uint16) prop.forceInteger();
   }
   else if( propName == "second" ) {
      m_second = (uint16) prop.forceInteger();
   }
   else if( propName == "msec" ) {
      m_msec = (uint16) prop.forceInteger();
   }
}

UserData *TimeStamp::clone()
{
   return new TimeStamp( *this );
}

}

/* end of TimeStampapi.cpp */
