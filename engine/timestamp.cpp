/*
   FALCON - The Falcon Programming Language.
   FILE: TimeStamp.cpp

   Implementation of the non-system specific TimeStamp class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/timestamp.h>
#include <falcon/autocstring.h>
#include <falcon/processor.h>
#include <falcon/vmcontext.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *RFC_2822_days[] = { "Mon","Tue", "Wed","Thu","Fri","Sat","Sun" };

static const char *Full_days[] = { "Monday","Tuesday", "Wednesday","Thursday","Friday","Saturday","Sunday" };

static const char *RFC_2822_months[] = {
   "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug", "Sep","Oct","Nov","Dec" };

static const char *Full_months[] = {
   "January","February","March","April","May","June","July","August", "September","October","November","December" };


#define SECOND_PER_DAY (60*60*24)

namespace Falcon {


TimeStamp::TimeStamp()
{
   // date set to 0
   m_bChanged = true;
   m_displacement = 0;
   m_timezone = tz_UTC;
   m_dst = false;
}


TimeStamp::TimeStamp( const Date& date, bool localTime ):
   m_date( date )
{
   m_bChanged = true;
   m_dst = localTime ? getLocalDST() : false;
   m_timezone = localTime ? getLocalTimeZone() : tz_UTC;

   int16 h,m;
   getTZDisplacement(m_timezone, h, m);
   m_displacement = h*60 + m;
}


void TimeStamp::computeDateFields() const
{
   // do this only if the date is not already known.
   if( ! m_bChanged )
   {
      return;
   }

   m_bChanged = false;

   // first, calculate the date since epoch.
   int64 secs = m_date.seconds() + (m_displacement*60);
   int64 days_since_epoch = secs;
   days_since_epoch /= SECOND_PER_DAY;
   secs %= SECOND_PER_DAY;

   int64 year;
   int16 month = 0;
   int16 month_days[] = {31,28,31,30,31,30,31,31,30,31,30,31};

   if( m_date.seconds() > 0 || (m_date.seconds() == 0 && m_date.femtoseconds() >= 0))
   {
      year = 1970;
      while (days_since_epoch >= 365)
      {
         days_since_epoch -= isLeapYear(year) ? 366 : 365;
         ++year;
      }

      if (isLeapYear(year))
      {
         month_days[1] = 29;
      }

      while( days_since_epoch >= month_days[month])
      {
         days_since_epoch -= month_days[month];
         ++month;
      }

      m_day = static_cast<int16>(days_since_epoch + 1);
      m_month = month + 1;
      m_year = year;

      m_msec = static_cast<int16>(m_date.femtoseconds()/1000000000000LL);
   }
   else
   {
      m_msec = static_cast<int16>((Date::FEMTOSECONDS + m_date.femtoseconds())/Date::MILLISECOND_DIVIDER);
      secs = SECOND_PER_DAY-1 + secs;
      // roll over milliseconds?
      if ( m_msec == 1000 )
      {
         secs++;
         m_msec = 0;
      }

      // roll over day?
      if( secs == SECOND_PER_DAY )
      {
         days_since_epoch ++;
         secs = 0;
      }

      year = 1969;
      while (days_since_epoch <= -365)
      {
         days_since_epoch += isLeapYear(year) ? 366 : 365;
         --year;
      }

      if (isLeapYear(year))
      {
         month_days[1] = 29;
      }

      month = 11;
      while( days_since_epoch <= -month_days[month])
      {
         days_since_epoch += month_days[month];
         --month;
      }

      m_day = static_cast<int16>(month_days[month] + days_since_epoch);
      m_month = month + 1;
      m_year = year;
   }

   m_hour = static_cast<int16>(secs/(60*60));
   secs %= (60*60);
   m_minute = static_cast<int16>(secs / 60);
   secs %= 60;
   m_second = static_cast<int16>(secs);

}


void TimeStamp::setCurrent(bool bLocal)
{
   m_bChanged = true;
   m_date.setCurrent();
   if ( bLocal )
   {
      timeZone(getLocalTimeZone());
   }

}


void TimeStamp::copy( const TimeStamp &ts )
{
   if( ! ts.m_bChanged )
   {
      m_year = ts.m_year;
      m_month = ts.m_month;
      m_day = ts.m_day;
      m_hour = ts.m_hour;
      m_minute = ts.m_minute;
      m_second = ts.m_second;
      m_msec = ts.m_msec;
   }

   m_bChanged = ts.m_bChanged;
   m_timezone = ts.m_timezone;
   m_displacement = ts.m_displacement;
   m_date = ts.m_date;
}


const char *TimeStamp::getRFC2822_ZoneName( TimeZone tz, bool bSemantic, bool bSaving )
{
   if ( tz == tz_local )
   {
      tz = getLocalTimeZone();
   }

   switch( tz )
   {
      case tz_local: return "+????";
      case tz_NONE: case tz_UTC: if ( bSemantic ) return "GMT"; return "+0000";
      case tz_UTC_E_1: if ( bSemantic && !bSaving) return "CET"; return "+0100";
      case tz_UTC_E_2: if ( bSemantic && bSaving) return "CET"; return "+0200";
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

      case tz_NFT: if ( bSemantic ) return "NFT"; return "+1130";
      case tz_ACDT: if ( bSemantic ) return "ACDT"; return "+1030";
      case tz_ACST: if ( bSemantic ) return "ACST"; return "+0930";
      case tz_HAT: if ( bSemantic ) return "HAT"; return "-0230";
      case tz_NST: if ( bSemantic ) return "NST"; return "-0330";
   }
   return "+????";
}


TimeStamp::TimeZone TimeStamp::getRFC2822_Zone( const char *csZoneName )
{
   if( strncmp( "UT", csZoneName, 2 ) == 0 ||
      strncmp( "UTC", csZoneName, 3 ) == 0 ||
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


bool TimeStamp::toRFC2822( String &target, bool bSemantic, bool bDst ) const
{
   computeDateFields();

   target.append( getRFC2822_WeekDayName( dayOfWeek() ) );
   target.append( ',' );
   target.append( ' ' );

   target.writeNumber( (int64) m_day, "%02d" );

   target.append( ' ' );
   target.append( getRFC2822_MonthName( m_month ) );
   target.append( ' ' );
   if ( m_year < 0 ) {
      target.append( "0000" );
   }
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
      tz = getLocalTimeZone();
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

   int64 year;
   int16 month;
   int16 day;
   int16 hour;
   int16 minute;
   int16 second;

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
   day = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 3)
      return false;
   month = getRFC2822_Month( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 4)
      return false;
   year = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ':' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   hour = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ':' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   minute = atoi( pos );

   mon++;
   pos = mon;
   while( *mon != 0 && *mon != ' ' ) mon++;
   if ( *mon == 0 || (mon - pos) != 2 )
      return false;
   second = atoi( pos );

   mon++;
   TimeZone tz = getRFC2822_Zone( mon );
   if( tz == tz_NONE )
   {
      return false;
   }

   if( ! target.set(year,month,day,hour,minute,second,0,tz) )
   {
      return false;
   }

   return true;
}



int16 TimeStamp::dayOfYear(int64 year, int16 month, int16 day)
{
   int days = 0;
   // add all the previous days
   switch( month ) {
      case 12: days += 30;
      /* no break */
      case 11: days += 31;
      /* no break */
      case 10: days += 30;
      /* no break */
      case 9: days += 31;
      /* no break */
      case 8: days += 31;
      /* no break */
      case 7: days += 30;
      /* no break */
      case 6: days += 31;
      /* no break */
      case 5: days += 30;
      /* no break */
      case 4: days += 31;
      /* no break */
      case 3: days += 28; if ( isLeapYear(year) ) days ++;
      /* no break */
      case 2: days += 31;
      /* no break */
   }
   days += day;

   return days;
}



int16 TimeStamp::getDaysOfMonth( int16 month, int64 year )
{
   switch( month ) {
      case 1: return 31;
      case 2: return isLeapYear(year)? 29 : 28;
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


int16 TimeStamp::dayOfWeek(int64 year, int16 month, int16 day)
{
   // consider 1700 the epoch
   if ( year < 1700 )
      return -1;

   // compute days since epoch.
   int32 nyears = static_cast<int32>(year - 1700);
   int32 nday = nyears * 365;

   // add leap years (up to the previous year. This year is  computed in dayOfYear()
   if( year > 1700 )
      nday += ((nyears-1) / 4) - ((nyears-1) / 100) + ((nyears-1) / 400);

   // add day of the year.
   nday += dayOfYear(year, month, day);

   // add epoch (1/1/1700 was a Friday)
   nday += 4;

   nday %= 7;
   return nday;
}


int16 TimeStamp::weekOfYear(int64 year, int16 month, int16 day, bool iso8601_2000 )
{
   int16 doy = dayOfYear(year, month, day);
   int16 week;
   // adjust for iso?
   if( iso8601_2000 )
   {
      week = (doy / 7)+1;
      week += static_cast<int16>(adjust_iso8601_2000(year, month, day));
      if( week == 0 )
      {
         week = 53;
      }
   }
   else
   {
      // compute the first monday
      int firstDay = dayOfYear(year, month, 1);
      int day = 1;
      while( firstDay %7 != 0 )
      {
         firstDay++;
         day++;
      }
      // week 0 is 1 -> 1st monday.
      week = (doy + 7 - day) / 7;
   }

   return week;
}


void TimeStamp::add( int32 days, int32 hours, int32 mins, int32 secs, int32 msecs )
{
   int64 seconds = days * SECOND_PER_DAY + hours*60*60 + mins *60 * secs;
   seconds *= 1000;
   seconds += msecs;
   if( seconds != 0 )
   {
      m_bChanged = true;
      m_date.addMilliseconds(seconds);
   }
}

void TimeStamp::add( int64 msecs )
{
   if( msecs != 0 )
   {
      m_bChanged = true;
      m_date.addMilliseconds(msecs);
   }
}



int64 TimeStamp::compare( const TimeStamp &ts ) const
{
   int64 dist = m_date.seconds() + (m_displacement*60) - ts.m_date.seconds() - (ts.m_displacement*60);
   if( dist == 0 )
   {
      dist = m_date.femtoseconds() - ts.m_date.femtoseconds();
   }

   return dist;
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
      tz = getLocalTimeZone();
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
   uint32 allocated = 25 >= FALCON_STRING_ALLOCATION_BLOCK ? 25 : FALCON_STRING_ALLOCATION_BLOCK;
   target.size(0);
   target.manipulator( &csh::handler_buffer );
   target.reserve(allocated);
   
   computeDateFields();
   sprintf( (char *)target.getRawStorage(), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
      (int32)m_year, m_month, m_day, m_hour, m_minute, m_second, m_msec );
   target.size(23);
}

bool TimeStamp::strftime( String &target, const String &fmt, length_t* posFail ) const
{
   computeDateFields();

   target.size(0);
   // a good guess.
   target.reserve(fmt.length()*2);

   // let's use a simple state machine
   length_t pos = 0;
   length_t len = fmt.length();
   while( pos+1 < len )
   {
      int32 chr = fmt.getCharAt(pos);
      bool localeName = false;
      bool localeNum = false;

      // check escape char...
      if( chr == (int32) '%' )
      {
         pos++;
         chr = fmt.getCharAt(pos);
         // type specifier?
         // E is locale names
         if( chr == 'E' )
         {
            ++pos;
            if( pos == len )
            {
               if( posFail != 0 ) *posFail = pos;
               return false;
            }

            chr = fmt.getCharAt(pos);
            if( chr != 'c' && chr != 'C' && chr != 'x' && chr != 'X' && chr !=  'y' && chr != 'Y' )
            {
               if( posFail != 0 ) *posFail = pos;
               return false;
            }

            localeName = true;
        }
        // O is locale numbers
        else if( chr == 'O' )
        {
           ++pos;
           if( pos == len )
           {
              if( posFail != 0 ) *posFail = pos;
              return false;
           }

           chr = fmt.getCharAt(pos);

           if( chr != 'd' && chr != 'e' && chr != 'H' && chr != 'I' && chr != 'm' &&
               chr != 'M' && chr != 'S' && chr != 'u' && chr != 'U' && chr != 'V' &&
               chr != 'w' && chr != 'W' && chr != 'y' )
           {
              if( posFail != 0 ) *posFail = pos;
              return false;
           }

           localeNum = true;
        }


         if( chr != '%')
         {
            String temp;
            if( ! strftimeChar( chr, temp ) )
            {
               if( posFail != 0 ) *posFail = pos;
               return false;
            }

            if( localeName || localeNum )
            {
               Processor *proc = Processor::currentProcessor();
               if( proc == 0 )
               {
                  if( posFail != 0 ) *posFail = pos;
                  return false;
               }

               // Numbers require single cipher translation.
               String temp2;
               if( localeNum )
               {
                  String cipher = "#TimeStamp:: ";
                  for( length_t i = 0; i < temp.length(); ++ i)
                  {
                     cipher.setCharAt(12, temp.getCharAt(i));
                     proc->currentContext()->process()->getTranslation(cipher, temp2);
                     target.append(temp2);
                     temp2.size(0); // just in case
                  }
               }
               else {
                  //proc->currentContext()->process()->getTranslation(temp, temp2);
                  //target.append(temp2);
                  target.append(temp);
               }
            }
            else
            {
               target.append(temp);
            }
         }
         else {
            target.append('%');
         }
      }
      else {
         target.append(chr);
      }

      pos++;
   }

   if( pos + 1 < len )
   {
      // last unparsed char.
      target.append(fmt.getCharAt(pos+1));
   }

   if( posFail != 0 ) *posFail = (length_t)-1;
   return true;
}


inline bool inner_translate( const String& source, String& target )
{
   Processor *proc = Processor::currentProcessor();
   if( proc != 0 )
   {
      return proc->currentContext()->process()->getTranslation(source, target);
   }
   else {
      target = source;
      return false;
   }
}


bool TimeStamp::strftimeChar( int32 chr, String &target ) const
{
   String temp;
   switch (chr)
   {
   case 'a':
      fassert( m_day >= 1 && m_day <= 7);
      temp = RFC_2822_days[m_day-1];
      inner_translate( temp, target );
      break;

   case 'A':
      fassert( m_day >= 1 && m_day <= 7);
      temp = Full_days[m_day-1];
      inner_translate( temp, target );
      break;

   case 'h':
   case 'b':
      fassert( m_month >= 1 && m_month <= 12);
      temp = RFC_2822_months[m_month-1];
      inner_translate( temp, target );
      break;

   case 'B':
      fassert( m_month >= 1 && m_month <= 12);
      temp = Full_months[m_day-1];
      inner_translate( temp, target );
      break;

   case 'c':
      //Date and time representation *   Thu Aug 23 14:55:02 2001
      {
         String format = "%a %b %d %T %Y";
         // allow an alternative format
         Processor *proc = Processor::currentProcessor();
         if( proc != 0 )
         {
            String tmpfmt;
            if( proc->currentContext()->process()->getTranslation("#TimeStamp::datetime", tmpfmt) )
            {
              format = tmpfmt;
            }
         }
         if (!strftime(target, format, 0) )
         {
            return false;
         }

      }
      break;

   case 'C':
      // century
      {
         int64 century = m_year/100;
         if( century < 10 )
            target.append('0');
         target.N(century);
      }
      break;

   case 'd':
      // day
      {
         if( m_day < 10 )
            target.append('0');
         target.N(m_day);
      }
      break;

   case 'D':
      // %m/%d/%y
      if( ! strftime(target,"%m/%d/%y",0) )
         return false;
      break;

   case 'e':
      {
         if( m_day < 10 )
            target.append(' ');
         target.N(m_day);
      }
      break;

   case 'F':
      // %Y-%m-%d
      if( ! strftime(target,"%Y-%m-%d",0) )
         return false;
      break;

   case 'g':
   case 'G':
      {
         int64 year = static_cast<int64>(adjust_iso8601_2000(m_year, m_month, m_day) + m_year);

         if( chr == 'g' )
         {
            year /= 100;
            if( year < 10 )
               target.append('0');
         }

         target.N(year);
      }
      break;

   case 'H':
      if( m_hour < 10 )
         target.append('0');
      target.N(m_hour);
      break;

   case 'i':
      toRFC2822(target,false, false);
      break;

   case 'I':
      // our in range 1-12
      {
         int16 hour = m_hour % 12;
         if( hour == 0 ) hour = 12;
         if( hour < 10 )
            target.append('0');
         target.N(hour);
         break;
      }
      break;

   case 'j':
      // dow
      {
         int16 dow = dayOfYear();
         if( dow < 100 ) target.append('0');
         if( dow < 10 ) target.append('0');
         target.N(dow);
      }
      break;

   case 'm':
      if( m_month < 10 )
         target.append('0');
      target.N(m_month);
      break;

   case 'M':
     if( m_minute < 10 ) target.append('0');
     target.N(m_minute);
     break;

   case 'n':
      target.append('\n');
      break;

   case 'p':
      {
         String marker = m_hour > 0 && m_hour <= 12 ? "AM" : "PM";
         if( ! inner_translate("#TimeStamp::" + marker, target ) )
         {
            target = marker;
         }
      }
      break;

   case 'q':
      target.append(m_msec);
      break;

   case 'Q':
      if( m_msec < 100 ) target.append('0');
      if( m_msec < 10 ) target.append('0');
      target.append(m_msec);
      break;

   case 'r':
      {
         String format;
         if (  ! inner_translate("#TimeStamp::12h", format ) )
         {
            format = "%I:%M:%S %p";
         }

         strftime(target, format, 0);
      }
      break;

   case 'R':
      strftime(target, "%H:%M", 0);
      break;

   case 'S':
      if( m_second < 10 )
         target.append('0');
      target.N(m_second);
      break;

   case 't':
      target.append('\t');
      break;

   case 'T':
      strftime(target, "%H:%M:%S", 0);
      break;

   case 'u':
      target.N( dayOfWeek()+1 );
      break;

   case 'U':
      {
         int16 woy = weekOfYear(false);
         // we calculate monday as first day of week, but %U starts from sunday.
         if( woy == 1 )
         {
            if( dayOfWeek() == 6) woy++;
         }
         else if( woy == 1 )
         {
            if( dayOfWeek() == 6) woy++;
         }

         if( woy < 10 ) target.append('0');
         target.N( woy );
      }
      break;

   case 'V':
      {
         int16 woy = weekOfYear(true);
         if( woy < 10 ) target.append('0');
         target.N( woy );
      }
      break;

   case 'w':
      target.N( (dayOfWeek()+1) % 7 );
      break;

   case 'W':
      {
         int16 woy = weekOfYear(false);
         if( woy < 10 ) target.append('0');
         target.N( woy );
      }
      break;

   case 'x':
      {
         String format;
         if( ! inner_translate("#TimeStamp::date_format", target ) )
         {
            format = "%Y-%m-%d";
         }
         strftime(target, format, 0);
      }
      break;

   case 'X':
      {
         String format;
         if( ! inner_translate("#TimeStamp::time_format", target ) )
         {
            format = "%H:%M:%S";
         }
         strftime(target, format, 0);
      }
      break;

   case 'y':
      {
         int year = m_year % 100;
         if( year < 10 ) target.append(year);
         target.N(year);
      }
      break;

   case 'Y':
      target.N(m_year);
      break;

   case 'z':
      {
         int16 h = 0 ,m = 0;
         getTZDisplacement(h,m);
         target.append(h < 0 ? '-' : '+');
         if( h < 10 ) target.append( '0' );
         target.N(h);
         if( m < 10 ) target.append( '0' );
         target.N(m);
      }
      break;

   case 'Z':
      if( m_timezone != tz_NONE )
      {
         target.append(getRFC2822_ZoneName(m_timezone, true, m_dst));
      }
      break;

   default:
      return false;
   }

   return true;
}


int16 TimeStamp::adjust_iso8601_2000( int64 year, int16 month, int16 day )
{
   if ( month == 1 )
   {
      if( day < 4 )
      {
         // 0 is monday...
         int32 dw = static_cast<int32>(dayOfWeek(year, (int16)month, (int16)day));
         // so, if current day
         if( (day - dw) < 0 )
         {
            return -1;
         }
      }
   }
   else if( month == 12 )
   {
      if( day > 28 )
      {
         int32 dw = static_cast<int32>(dayOfWeek(year, month, day));
         if ( (31-dw) > 28 )
         {
            return +1;
         }
      }
   }

   return 0;
}



bool TimeStamp::year(int16 value)
{
   computeDateFields();
   return set( value, m_month, m_day, m_hour, m_minute, m_second, m_msec, m_timezone );
}

bool TimeStamp::month(int16 value)
{
   computeDateFields();
   return set( m_year, value, m_day, m_hour, m_minute, m_second, m_msec, m_timezone );
}


bool TimeStamp::day(int16 value)
{
   computeDateFields();
   return set( m_year, m_month, value, m_hour, m_minute, m_second, m_msec, m_timezone );
}


bool TimeStamp::hour(int16 value)
{
   computeDateFields();
   return set( m_year, m_month, m_day, value, m_minute, m_second, m_msec, m_timezone );
}


bool TimeStamp::minute(int16 value)
{
   computeDateFields();
   return set( m_year, m_month, m_day, m_hour, value, m_second, m_msec, m_timezone );
}


bool TimeStamp::second(int16 value)
{
   computeDateFields();
   return set( m_year, m_month, m_day, m_hour, m_minute, value, m_msec, m_timezone );
}


bool TimeStamp::msec(int16 value)
{
   computeDateFields();
   return set( m_year, m_month, m_day, m_hour, m_minute, m_second, value, m_timezone );
}



bool TimeStamp::set( int64 y, int16 M, int16 d, int16 h, int16 m,
         int16 s, int16 ms, TimeZone tz )
{
   int16 hdisp = 0, mdisp = 0;
   getTZDisplacement(tz, hdisp, mdisp);
   return set( y, M, d, h, m, s, ms, hdisp*60 + mdisp );
}


bool TimeStamp::set( int64 y, int16 M, int16 d, int16 h, int16 m,
         int16 s, int16 ms, int16 disp )
{
   if( M < 1 || M > 12 || d < 1 || d > getDaysOfMonth(M,y) )
   {
      return false;
   }

   int64 baseMs = 0;

   // now the fun part, calculate the year displacement.
   if( y >= 1970 )
   {
      int64 days = d-1;
      int64 year = 1970;
      while( year < y )
      {
         days += isLeapYear(year) ? 366: 365;
         ++year;
      }

      int month = 1;
      while( month < M )
      {
         days += getDaysOfMonth(month,y);
         month++;
      }

      baseMs = s+ m*60 + h*3600 + days * SECOND_PER_DAY;
      baseMs *= 1000;
      baseMs += ms;
   }
   else
   {
      int64 days = 0;
      int64 year = 1969;
      while( year > y )
      {
         days -= isLeapYear(year) ? 366: 365;
         --year;
      }

      int month = 12;
      while( month > M )
      {
         days -= getDaysOfMonth(month,y);
         month--;
      }

      days -= getDaysOfMonth(M,year) - d; // would be -d+1 but...
      baseMs = -(SECOND_PER_DAY*1000) + (s*1000+ m*60000 + h*3600000 + ms);  // we must consider the part of the day we're adding.
      baseMs += days*SECOND_PER_DAY*1000;
   }

   m_date.fromMilliseconds(baseMs);

   m_year = y;
   m_month = M;
   m_day = d;
   m_hour = h;
   m_minute = m;
   m_second = s;
   m_msec = ms;
   m_timezone = displacementToTZ(disp);
   m_displacement = disp;

   m_bChanged = false;

   return true;
}


bool TimeStamp::setTime( int16 h, int16 m, int16 s, int16 ms, TimeZone tz )
{
   computeDateFields();
   return set( m_year, m_month, m_day, h, m, s, ms, tz );
}


void TimeStamp::set( const Date& date )
{
   m_bChanged = true;
   m_date = date;
}


void TimeStamp::changeTimeZone( TimeZone tz )
{
   TimeZone tz1 = tz;
   if ( tz1 == tz_local )
   {
      tz1 = getLocalTimeZone();
   }

   // no shift?
   if ( tz1 == m_timezone || m_timezone == tz_local )
   {
      return;
   }

   // get the relative total shift.
   int16 newHour=0, newMin=0;
   getTZDisplacement( tz1, newHour, newMin );

   // bring displacement back into the date, to keep it constant with the new displacement.
   int16 newDisp = newHour*60 + newMin;
   m_date.addSeconds( (m_displacement * 60) - (newDisp * 60));
   m_displacement = newDisp;
   m_timezone = tz;
}


void TimeStamp::msSinceEpoch( int64 v )
{
   int64 disp = m_displacement;
   disp *= 60000;
   m_date.fromMilliseconds(v - disp);
}


int64 TimeStamp::msSinceEpoch() const
{
   int64 ms = m_displacement;
   ms *= 60000;
   ms += m_date.toMilliseconds();
   return ms;
}


bool TimeStamp::changeDisplacement(int16 value)
{
   if( value < -7200 || value > 7200 )
   {
      return false;
   }

   m_date.addSeconds( (m_displacement * 60) - (value * 60));
   m_displacement = value;
   m_timezone = displacementToTZ(value);
   m_bChanged = true;
   return true;
}


bool TimeStamp::displacement(int16 value)
{
   if( value < -7200 || value > 7200 )
   {
      return false;
   }

   m_displacement = value;
   m_timezone = displacementToTZ(value);
   m_bChanged = true;
   return true;
}


void TimeStamp::timeZone(TimeZone tz)
{
   TimeZone tz1 = tz;
   if ( tz1 == tz_local )
   {
      tz1 = getLocalTimeZone();
   }

   // no shift?
   if ( tz1 == m_timezone || m_timezone == tz_local )
   {
      return;
   }

   // get the relative total shift.
   int16 newHour=0, newMin=0;
   getTZDisplacement( tz1, newHour, newMin );
   m_displacement = newHour*60 + newMin;
   m_timezone = tz;  // keep tz_local if it was set
   m_bChanged = true;
}


TimeStamp::TimeZone TimeStamp::displacementToTZ( int16 mindisp )
{
   switch( mindisp )
   {
      case 0: return tz_UTC;

      case 1 * 60: return tz_UTC_E_1;
      case 2 * 60: return tz_UTC_E_2;
      case 3 * 60: return tz_UTC_E_3;
      case 4 * 60: return tz_UTC_E_4;
      case 5 * 60: return tz_UTC_E_5;
      case 6 * 60: return tz_UTC_E_6;
      case 7 * 60: return tz_UTC_E_7;
      case 8 * 60: return tz_UTC_E_8;
      case 9 * 60: return tz_UTC_E_9;
      case 10 * 60: return tz_UTC_E_10;
      case 11 * 60: return tz_UTC_E_11;
      case 12 * 60: return tz_UTC_E_12;

      case -1 * 60: return tz_UTC_W_1;
      case -2 * 60: return tz_UTC_W_2;
      case -3 * 60: return tz_UTC_W_3;
      case -4 * 60: return tz_UTC_W_4;
      case -5 * 60: return tz_UTC_W_5;
      case -6 * 60: return tz_UTC_W_6;
      case -7 * 60: return tz_UTC_W_7;
      case -8 * 60: return tz_UTC_W_8;
      case -9 * 60: return tz_UTC_W_9;
      case -10 * 60: return tz_UTC_W_10;
      case -11 * 60: return tz_UTC_W_11;
      case -12 * 60: return tz_UTC_W_12;

      case 10 * 60+30: return tz_ACDT;
      case 11 * 60+30: return tz_NFT;
      case -2 * 60-30: return tz_HAT;
      case -3 * 60-30: return tz_NST;
   }

   return tz_NONE;
}


TimeStamp *TimeStamp::clone() const
{
   return new TimeStamp( *this );
}


void TimeStamp::store( DataWriter* dw ) const
{
   dw->write((char) m_timezone);
   dw->write(m_displacement);
   dw->write(m_dst);
   dw->write(m_date);
}


void TimeStamp::restore( DataReader* dr )
{
   char tz;
   int16 dist;
   bool dst;
   dr->read(tz);
   dr->read(dist);
   dr->read(dst);
   dr->read(m_date);

   m_timezone = (TimeZone) tz;
   m_displacement = dist;
   m_dst = dst;
   m_bChanged = true;
}


void TimeStamp::currentTime()
{
   setCurrent(true);
}


void TimeStamp::setDST( bool dst )
{
   // nothing to do?
   if( dst == m_dst )
   {
      return;
   }

   if( dst )
   {
      m_displacement += 60;
   }
   else {
      m_displacement -= 60;
   }

   m_dst = dst;
   m_timezone = displacementToTZ(m_displacement);
   m_bChanged = true;
}


void TimeStamp::changeDST( bool dst )
{
   // nothing to do?
   if( dst == m_dst )
   {
      return;
   }

   m_dst = dst;
   changeDisplacement( m_displacement + (dst? +60 : -60) );
}


}

/* end of TimeStampapi.cpp */
