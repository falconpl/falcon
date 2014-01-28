/*
   FALCON - The Falcon Programming Language.
   FILE: timestamp.h

   Multiplatform date and time description.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Multiplatform date and time description.
*/

#ifndef _FALCON_TIMESTAMP_H_
#define _FALCON_TIMESTAMP_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/refpointer.h>
#include <falcon/date.h>

namespace Falcon {

class String;
class DataReader;
class DataWriter;

/** TimeStamp class.

 This class is both used as a system independent time accounting
   object and as a internal object for the TimeStamp falcon core
   object.

 The TimeStamp class can indicate a precise moment in time (up to a millisecond
 precision) or a time spam, that is, a set of years, months, days, hours, minutes
 seconds and fractions that separate two times.

*/
class FALCON_DYN_CLASS TimeStamp
{
public:
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

   /** Creates an empty date.
    *
    * The date points at epoch (1/1/1970, 00:00 GMT).
    *
    * To create a date already set at current time, use TimeStamp(Date::current()),
    * and then eventually shift the timezone.
    */
   TimeStamp();

   /** Creates a date from a given date field.
    * \param date Date in GMT
    * \param localTime set to true to transform the date in localtime.
    *
    * The input date is considered to be in GMT; if localTime is set to true,
    * the timestamp will be shifted.
    */
   TimeStamp( const Date& date, bool localTime = false );

   TimeStamp( const TimeStamp &ts )
   {
      copy( ts );
   }

   virtual ~TimeStamp() {}

   /** Set this timestamp as the system current time.
    \param bLocal if true (default), the current time is set to the current timezone.

    If bLocal is false, the current time is set in GMT.
   */
   void setCurrent( bool bLocal = true );

   void copy( const TimeStamp &ts );




   /** Return the local timezone. */
   static TimeZone getLocalTimeZone();

   /** Return true if DST is in effect in the local timezone. */
   static bool getLocalDST();

   /**
    * Gets the date represented in this timestamp.
    */
   const Date& date() const { return m_date; }

   /**
    * Gets the date represented in this timestamp.
    *
    * The non-const version of this method will set the changed field
    * and force to recalculate the timestamp fields when required.
    */
   Date& date() { m_bChanged = true; return m_date; }

   /** Return a RCF2822 timezone name.
      \param tz The timezone.
      \param bSemantic return semantic zone names instead of + displacements when available.
      \param bDst Get the DST version of the semantic zone.
      \return the zone name.
   */
   static const char *getRFC2822_ZoneName( TimeZone tz, bool bSemantic=false, bool bDst=false );

   /** Return a timezone given a RCF2822 timezone name.
      \param csZoneName The timezone name.
      \return the zone, or tz_NONE if the zone didn't parse succesfully.
   */
   static TimeZone getRFC2822_Zone( const char *csZoneName );

   /** Get a timezone displacement
      \param tz the timezone.
      \param hours hours the displacement in hours
      \param minutes minutes the displacement in minutes
   */
   static void getTZDisplacement( TimeZone tz, int16 &hours, int16 &minutes );

   /** Gets a RFC 2822 timestamp compliant weekday name.
      \return weekday name.
   */
   static const char *getRFC2822_WeekDayName( int16 wd );

   /** Gets a RFC 2822 timestamp compliant month name.
      \return Month name.
   */
   static const char *getRFC2822_MonthName( int16 wd );

   /** Return numeric weekday from a RFC2822 format weekday name.
      \return -1 if the name is not valid, 0-6 otherwise (Monday being 0).
   */
   static int16 getRFC2822_WeekDay( const char *name );

   /** Return numeric month from a RFC2822 format month name.
      \return -1 if the name is not valid, 1-12 otherwise (january being 1).
   */
   static int16 getRFC2822_Month( const char *name );

   /** Convert this timestamp to RFC2822 format.
      \param target The string that will receive the converted date, or "?" in case it doesn't work.
      \param bSemantic return semantic zone names instead of + displacements when available.
      \param bDst Get the DST version of the semantic zone.
      \return false if the date is invalid.
   */
   bool toRFC2822( String &target, bool bSemantic=false, bool bDst=false ) const;

   /** Convert this timestamp to RFC2822 format.
      \param bSemantic return semantic zone names instead of + displacements when available.
      \param bDst Get the DST version of the semantic zone.
      \return The string converted.
   */
   String toRFC2822( bool bSemantic=false, bool bDst=false ) const
   {
      String temp(32);
      toRFC2822( temp, bSemantic, bDst );
      return temp;
   }

   /** Parse a RFC2822 date format and configure the given timestamp. */
   static bool fromRFC2822( TimeStamp &target, const String &source );
   /** Parse a RFC2822 date format and configure the given timestamp. */
   static bool fromRFC2822( TimeStamp &target, const char *source );

   /**
    * Equivalent to setCurrent() on local timezone.
    */
   void currentTime();
   bool isLeapYear() const
   {
      return isLeapYear( m_year );
   }

   static bool isLeapYear( int64 year )
   {
      if ( year % 100 == 0 ) {
         // centuries that can be divided by 400 are leap
         // others are not.
         return ( year % 400 == 0 );
      }

      // all other divisible by 4 years are leap
      return ( year % 4 == 0 );
   }

   static int16 dayOfYear(int64 year, int16 month, int16 day);

   int16 dayOfYear() const
   {
      computeDateFields();
      return dayOfYear(m_year, m_month, m_day );
   }


   /** Gets the day of week.
    \return the day of the week, in range 0-6 (0 is monday).
      Week starting on monday, 0 based.
   */
   static int16 dayOfWeek(int64 year, int16 month, int16 day);

   int16 dayOfWeek() const
   {
      computeDateFields();
      return dayOfWeek( m_year, m_month, m_day );
   }

   /**
    * Returns the week of the year for the given date.
    * \param year the date year
    * \param month the date month (1-12)
    * \param day the date day (1-31)
    * \param iso8601_2000 if true, adjust the week as recommended by iso8601_2000.
    *
    * \see adjust_iso8601_2000
    */
   static int16 weekOfYear(int64 year, int16 month, int16 day, bool iso8601_2000 = false );

   /**
    * Returns the week of the year for this date.
    * \param iso8601_2000 if true, adjust the week as recommended by iso8601_2000.
    *
    * \see adjust_iso8601_2000
    */
   int16 weekOfYear( bool iso8601_2000 = false ) const
   {
      computeDateFields();
      return weekOfYear( m_year, m_month, m_day, iso8601_2000 );
   }

   /**
    * Adds a given number of days, hours, minutes, seconds and milliseconds to the timestamp.
    *
    * \note To add a number of milliseconds to the timestamp, use the date() field and
    * add the milliseconds to that.
    */
   void add( int32 days, int32 hours=0, int32 mins=0, int32 secs=0, int32 msecs=0 );
   void add( int64 msecs );

   int64 compare( const TimeStamp &ts ) const;

   /** Performs a simple conversion to staring.
    * \target the string where to store the result
    *
    * This renders the timestamp in format "YYYY-MM-DD HH:mm:SS.sss" where "sss" is milliseconds.
    * \see strftime
    */
   void toString( String &target ) const;

   /**
    Converts the timestamp through a format string.
    \param target Where to store the string
    \param fmt The string format for rendering
    \param posFail If non zero, will receive the position where the parsing failed in case of error
    \return true on success, false on error.

    This function renders the timestamp using a format that is similar to that of the POSIX strftime applied on
    C time stamp structures.

    Falcon doesn't support C locale functions natively (they are provided through a separate module), but
    internationalization is supported in this function through Process::setTranslation() system.

   The following conversion specifications are supported:

    - %a: Replaced by the locale's abbreviated weekday name. By default, it's English 3 character names,
          that can be overridden by setting the translations for "Mon","Tue", "Wed","Thu","Fri","Sat","Sun".
    - %A: Replaced by the locale's full weekday name. By default, it's English day names, that can be overridden
          by setting the translations for "Monday","Tuesday", "Wednesday","Thursday","Friday","Saturday","Sunday".
    - %b: Replaced by the locale's abbreviated month name. By default, it's English 3 characters month names,
            that can be overridden by setting the translations for
            "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug", "Sep","Oct","Nov","Dec".
    - %B: Replaced by the locale's full month name. By default, it's English month names,
            that can be overridden by setting the translations for
            "January","February","March","April","May","June","July","August", "September","October","November","December".
    - %c: Replaced by the locale's appropriate date and time representation.
          By default, it's "%a %b %d %T %Y", but can be overridden by setting the "#TimeStamp::datetime" name.
    - %C: Replaced by the year divided by 100 and truncated to an integer, as a decimal number [00,99].
    - %d: Replaced by the day of the month as a decimal number [01,31]. [ tm_mday]
    - %D: Equivalent to "%m/%d/%y".
    - %e: Replaced by the day of the month as a decimal number [1,31]; a single digit is preceded by a space.
    - %F: Equivalent to "%Y-%m-%d" (the ISO 8601:2000 standard date format).
    - %g: Replaced by the last 2 digits of the week-based year (see below) as a decimal number [00,99].
    - %G: Replaced by the week-based year (see below) as a decimal number (for example, 1977).
    - %h: Equivalent to %b.
    - %H: Replaced by the hour (24-hour clock) as a decimal number [00,23].
    - %I: Replaced by the hour (12-hour clock) as a decimal number [01,12].
    - %j: Replaced by the day of the year as a decimal number [001,366].
    - %m: Replaced by the month as a decimal number [01,12].
    - %M: Replaced by the minute as a decimal number [00,59].
    - %n: Replaced by a <newline>.
    - %p: Replaced by the locale's equivalent of either a.m. or p.m. Defaults to "AM" and "PM", and can be
          overridden by setting the translations for "#TimeStamp::AM" and "#TimeStamp::PM"
    - %r: Replaced by the time in a.m. and p.m. notation; by default, it's equivalent to "%I:%M:%S %p",
          but can be overridden by setting the transation for "#TimeStamp::12h".
    - %R: Replaced by the time in 24-hour notation "%H:%M".
    - %S: Replaced by the second as a decimal number [00,60].
    - %t: Replaced by a <tab>.
    - %T: Replaced by the time ( %H : %M : %S ). [ tm_hour, tm_min, tm_sec]
    - %u: Replaced by the weekday as a decimal number [1,7], with 1 representing Monday.
    - %U: Replaced by the week number of the year as a decimal number [00,53]. The first Sunday of January is the
          first day of week 1; days in the new year before this are in week 0.
    - %V: Replaced by the week number of the year (Monday as the first day of the week) as a decimal number [01,53].
          If the week containing 1 January has four or more days in the new year, then it is considered week 1.
          Otherwise, it is the last week of the previous year, and the next week is week 1.
          Both January 4th and the first Thursday of January are always in week 1.
    - %w: Replaced by the weekday as a decimal number [0,6], with 0 representing Sunday.
    - %W: Replaced by the week number of the year as a decimal number [00,53].
          The first Monday of January is the first day of week 1; days in the new year before this are in week 0.
    - %x: Replaced by the locale's appropriate date representation. By default, it's "%Y-%m-%d", but it can be
          overridden by setting the translation for "#TimeStamp::date_format".
    - %X: Replaced by the locale's appropriate time representation. By default, it's "%H:%M:%S", but it can be
          overridden by setting the translation for "#TimeStamp::time_format".
    - %y: Replaced by the last two digits of the year as a decimal number [00,99].
    - %Y: Replaced by the year as a decimal number (for example, 1997).
    - %z: Replaced by the offset from UTC in the ISO 8601:2000 standard format ( +hhmm or -hhmm ),
          or by no characters if no timezone is determinable. For example, "-0430" means 4 hours 30 minutes
          behind UTC (west of Greenwich).
    - %Z: Replaced by the timezone name or abbreviation, or by no bytes if no timezone information exists.
    - %%: Replaced by %.

    Some of the formats can be specifically translated using the 'E' and 'O' type specifiers
    - E  Uses the locale's alternative representation %Ec %EC %Ex %EX %Ey %EY. It's actually ignored.
    - O  Uses the locale's alternative numeric symbols  %Od %Oe %OH %OI %Om %OM %OS %Ou %OU %OV %Ow %OW %Oy
         To set locale symbols for the ciphers in place of the arabic cypers, use set the translation
          "#TimeStamp::N", where N is 0 to 9;

    Falcon also provides the following extension with respect to the standard formats:
    - %i: Replaced by the international RFC2822 representation (HTTP and MIME headers format).
    - %q: Replaced by the milliseconds, not padded.
    - %Q: Replaced by the milliseconds, padded with zero, in range [000-999].
   */

   bool strftime( String &target, const String &fmt, length_t* posFail ) const;
   bool strftimeChar( int32 chr, String &target ) const;

   void getTZDisplacement( int16 &hours, int16 &minutes ) const;

   bool operator ==( const TimeStamp &ts ) const { return this->date() == ts.date(); }
   bool operator !=( const TimeStamp &ts ) const { return this->date() != ts.date(); }
   bool operator <( const TimeStamp &ts ) const  { return this->date() <  ts.date(); }
   bool operator >( const TimeStamp &ts ) const  { return this->date() >  ts.date(); }
   bool operator <=( const TimeStamp &ts ) const { return this->date() <= ts.date(); }
   bool operator >=( const TimeStamp &ts ) const { return this->date() >= ts.date(); }

   void store( DataWriter* dw ) const;
   void restore( DataReader* dr );
   virtual TimeStamp* clone() const;

   /**
    * Returns the days of the given months in the date year.
    */

   static int16 getDaysOfMonth( int16 month, int64 year );

   int64 year() const { computeDateFields(); return m_year; }
   int16 month() const { computeDateFields(); return m_month; }
   int16 day() const { computeDateFields(); return m_day; }
   int16 hour() const { computeDateFields(); return m_hour; }
   int16 minute() const { computeDateFields(); return m_minute; }
   int16 second() const { computeDateFields();  return m_second; }
   int16 msec() const { computeDateFields();return m_msec; }
   int16 displacement() const { return m_displacement; }
   TimeZone timeZone() const { return m_timezone; }
   bool isDST() const { return m_dst; }

   /** Sets the year of the current date.
    * \param value The year to be set.
    * \return true if setting the year leaves a valid date.
    *
    * The function might return false if the current day of the date is February the 29nt,
    * and the year that was set didn't have such day.
    */
   bool year(int16 value);

   /** Sets the month of the current date.
    * \param value The month to be set.
    * \return false if the month is outside 1-12 range, or if
    * the date would become invalid in setting that month.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool month(int16 value);

   /** Sets the day of the current date.
    * \param value The day to be set.
    * \return false if the month is outside 1-<current month days> range, or if
    * the date would become invalid in setting that month.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool day(int16 value);

   /** Sets the hour of the current date.
    * \param value The hour to be set.
    * \return false if the month is outside 0-23 range.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool hour(int16 value);

   /** Sets the minute of the current date.
    * \param value The minute to be set.
    * \return false if the minute is outside 0-59 range.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool minute(int16 value);

   /** Sets the second of the current date.
    * \param value The second to be set.
    * \return false if the second is outside 0-59 range.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool second(int16 value);

   /** Sets the second of the current date.
    * \param value The second to be set.
    * \return false if the month is outside 0-59 range.
    *
    * \note Do not use this function to set a whole date or time.
    *  Use set() to set the whole date.
    */
   bool msec(int16 value);

   /** Sets the displacement (disatance from GMT).
    * \param value The displacement to be set.
    * \return false if the displacement is outside -7200 : +7200 range.
    *
    * This method will also adjust the date timezone. If the displacement
    * corresponds to a known timezone, that timezone will be set, otherwise
    * the timezone will be set to NONE.
    *
    * The set displacement will cause the reported date to move accordingly,
    * so that if the time set is 22:00, and displacement is set to +30 minutes,
    * the date reported will be 22:30.
    *
    * To change the displacement without changing the reported time, use
    * changeDisplacement()
    */
   bool displacement(int16 value);

   /** Sets the displacement (disatance from GMT) while keeping the same day-time value.
    * \param value The displacement to be set.
    * \return false if the displacement is outside -7200 : +7200 range.
    *
    * This method will also adjust the date timezone. If the displacement
    * corresponds to a known timezone, that timezone will be set, otherwise
    * the timezone will be set to NONE.
    *
    * This method will not change the reported time, so that if the time
    * is currently 22:00 GMT, setting a displacement of +60 will cause
    * the time to be set as 22:00 GMT+1. To displace the reported time,
    * use the displacement() method.
    */
   bool changeDisplacement(int16 value);

   /** Sets the daylight saving time.
    * \param True to set DST on, false to disable it.
    *
    */
   void setDST( bool dst );

   /** Sets the daylight saving time status while keeping the same day-time value.
    * \param True to set DST on, false to disable it.
    *
    * This method will not change the reported time, so that if the time
    * is currently 22:00 GMT, setting a displacement of DST will cause
    * the time to be set as 22:00 GMT+1. To change the reported time, use
    * the setDST method.
    */
   void changeDST( bool dst );

   /** Sets the date.
    * \bool true if the date can be set, false if it's invalid.
    *
    */
   bool set( int64 y, int16 M, int16 d, int16 h, int16 m,
            int16 s, int16 ms, TimeZone tz );

   /** Sets the date.
    * \bool true if the date can be set, false if it's invalid.
    *
    */
   bool set( int64 y, int16 M, int16 d, int16 h=0, int16 m=0,
            int16 s=0, int16 ms = 0, int16 displacement=0 );

   /** Sets the time for the given date.
    * \bool true if the date can be set, false if its' invalid.
    */
   bool setTime( int16 h=0, int16 m=0, int16 s=0, int16 ms = 0, TimeZone tz=tz_NONE );

   /** Sets the date using a Date entity.
    * \bool true if the date can be set, false if its' invalid.
    */
   void set( const Date& date );

   /** Changes the timezone associated with this date.
    *
    * This will change the timezone, so that if you have a date in GMT,
    * and the timezone is set to GMT+2, the time will be moved 2 hours
    * forward. For instance, it the time is 20:31 GMT, and the new timezone
    * is GMT+2, the time will be set to 18:31 GMT+2.
    *
    * To change the timezone without changing the timestamp value,
    * use changeTimeZone().
    */
   void timeZone(TimeZone tz);

   /** Shifts this timestamp moving the old timezone into the new one.
    * \param tz new timezone.
    *
    * This method changes the timezone without altering the date and time
    * reported by this timestamp. For example, is the time is set to 22:31 GMT,
    * and this function is used to set the timezone to GMT+2, the time becomes
    * 22:31 GMT+2. To move the timezone and change the time relatively to GMT,
    * use the timeZone() method.
    */
   void changeTimeZone( TimeZone tz );

   void msSinceEpoch( int64 v );
   int64 msSinceEpoch() const;


   /** Returns the adjustment for week-based year (According with ISO8601:2000)
    * \param year the year of the date to be checked
    * \param month the month of the date to be checked
    * \param day the day of the date to be checked
    * \reutrn 0 if the date doesn't need to be adjusted, -1 if this week goes in the previous year,
    *       +1 if it goes in the next year.
    *
    * In this system, weeks begin on a Monday and week 1 of the year is the week that includes January 4th,
    * which is also the week that includes the first Thursday of the year, and is also the first week that
    * contains at least four days in the year.
    *
    * If the first Monday of January is the 2nd, 3rd, or 4th,
    *  the preceding days are part of the last week of the preceding year; thus, for Saturday 2nd January 1999,
    *  the year is 1998 and the week is 53.
    *
    * If December 29th, 30th, or 31st is a Monday, it and any following days are part of week 1 of the following year.
    *
    */
   static int16 adjust_iso8601_2000( int64 year, int16 month, int16 day );

   /** Return the timezone corresponding to the given displacement in minutes. */
   static TimeZone displacementToTZ( int16 mindisp );

   void gcMark(uint32 mark ) { m_gcMark = mark; }
   uint32 currentMark() const { return m_gcMark; }

private:

   mutable int64 m_year;
   mutable int16 m_month;
   mutable int16 m_day;
   mutable int16 m_hour;
   mutable int16 m_minute;
   mutable int16 m_second;
   mutable int16 m_msec;

   TimeZone m_timezone;
   int16 m_displacement;
   bool m_dst;

   mutable bool m_bChanged;

   // The real underlying date.
   Date m_date;

   uint32 m_gcMark;
   /**
    * Compute date fields from the stored date entity.
    */
   void computeDateFields() const;
};


}

#endif

/* end of timestamp.h */
