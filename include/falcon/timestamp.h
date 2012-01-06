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

#include <falcon/interrupt.h>

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


   int16 m_year;
   int16 m_month;
   int16 m_day;
   int16 m_hour;
   int16 m_minute;
   int16 m_second;
   int16 m_msec;
   TimeZone m_timezone;

   TimeStamp( int16 y=0, int16 M=0, int16 d=0, int16 h=0, int16 m=0,
               int16 s=0, int16 ms = 0, TimeZone tz=tz_NONE ):
         m_year( y ),
         m_month( M ),
         m_day( d ),
         m_hour( h ),
         m_minute( m ),
         m_second( s ),
         m_msec( ms ),
         m_timezone( tz )
   {}

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

   /** Return the local timezone. */
   static TimeZone getLocalTimeZone();

   /** Wait until the given timestamp expires.
    \param ts The timestamp to wait for.
    \param intr The wait might be interrupted signaling the interrupter.
    \return true if the wait was performed, false if the timestamp is in the past.
    \throw InterruptedError if the wait was interrupted.

    If the timestamp is in the past, no wait is actually performed.

    */
   static bool absoluteWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr );

   /** Wait until the given timestamp expires.
    \param ts The timestamp to wait for.
    \return true if the wait was performed, false if the timestamp is in the past.
    \throw InterruptedError if the wait was interrupted.

    If the timestamp is in the past, no wait is actually performed.

    */
   static bool absoluteWait( const TimeStamp &ts );


   /** Wait the required amount of years, months, days etc...
    \param ts Number of years, months etc. to wait for.
    \param intr If given the wait might be interrupted signaling the interrupter.
    \return true if the wait was performed, false if the timestamp is in the past.
    \throw InterruptedError if the wait was interrupted.

    If the timestamp is negative , no wait is actually performed.
   */
   static bool relativeWait( const TimeStamp &ts, ref_ptr<Interrupt>& intr );

   /** Wait the required amount of years, months, days etc...
    \param ts Number of years, months etc. to wait for.
    \return true if the wait was performed, false if the timestamp is in the past.
    \throw InterruptedError if the wait was interrupted.

    If the timestamp is negative , no wait is actually performed.
   */
   static bool relativeWait( const TimeStamp &ts );

   /** Initialize this timestamp using a system time structure.
    \param system_time A system-specific time structure that can be used to create
                       a timestamp.
    */
   void fromSystemTime( void* system_time );


   TimeStamp &operator = ( const TimeStamp &ts );

   TimeStamp &operator += ( const TimeStamp &ts )
   {
      add( ts );
      return *this;
   }

   TimeStamp &operator -= ( const TimeStamp &ts )
   {
      distance(ts);
      return *this;
   }

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

   /** Shifts this timestamp moving the old timezone into the new one. */
   void changeTimezone( TimeZone tz );

   void copy( const TimeStamp &ts );
   void currentTime();
   bool isValid() const;
   bool isLeapYear() const;
   int16 dayOfYear() const;

   /** Gets the day of week.
      Week starting on monday, 0 based.
   */
   int16 dayOfWeek() const;

   void add( const TimeStamp &ts );
   void add( int32 days, int32 hours=0, int32 mins=0, int32 secs=0, int32 msecs=0 );
   void distance( const TimeStamp &ts );
   int32 compare( const TimeStamp &ts ) const;
   void toString( String &target ) const;

   bool toString( String &target, const String &fmt ) const;

   void getTZDisplacement( int16 &hours, int16 &minutes ) const;

   bool operator ==( const TimeStamp &ts ) const { return this->compare( ts ) == 0; }
   bool operator !=( const TimeStamp &ts ) const { return this->compare( ts ) != 0; }
   bool operator <( const TimeStamp &ts ) const { return this->compare( ts ) < 0; }
   bool operator >( const TimeStamp &ts ) const { return this->compare( ts ) > 0; }
   bool operator <=( const TimeStamp &ts ) const { return this->compare( ts ) <= 0; }
   bool operator >=( const TimeStamp &ts ) const { return this->compare( ts ) >= 0; }

   void serialize( DataWriter* dw ) const;
   void read( DataReader* dr );
   static TimeStamp* deserialize( DataReader* dw );

   static TimeStamp* create();
   virtual TimeStamp* clone() const;

private:
   void rollOver(  bool onlyDays = false);
   int16 getDaysOfMonth( int16 month = -1 ) const;
};

inline TimeStamp operator + ( const TimeStamp &ts1, const TimeStamp &ts2 )
{
   TimeStamp ts3(ts1);
   ts3.add( ts2 );
   return ts3;
}

inline TimeStamp operator - ( const TimeStamp &ts1, const TimeStamp &ts2 )
{
   TimeStamp ts3(ts1);
   ts3.distance( ts2 );
   return ts3;
}

}

#endif

/* end of timestamp.h */
