/*
   FALCON - The Falcon Programming Language.
   FILE: timestamp.h

   Multiplatform date and time description.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio giu 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Multiplatform date and time description.
*/

#ifndef flc_timestamp_H
#define flc_timestamp_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/falcondata.h>
#include <falcon/time_sys.h>
#include <falcon/string.h>

namespace Falcon {

class String;

/** TimeStamp class.
   This class is both used as a system independent time accounting
   object and as a internal object for the TimeStamp falcon core
   object.
*/
class FALCON_DYN_CLASS TimeStamp: public FalconData
{
private:
   void rollOver(  bool onlyDays = false);
   int16 getDaysOfMonth( int16 month = -1 ) const;

public:

   int16 m_year;
   int16 m_month;
   int16 m_day;
   int16 m_hour;
   int16 m_minute;
   int16 m_second;
   int16 m_msec;
   TimeZone m_timezone;

public:
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

   TimeStamp( int64 lfmt )
   {
      fromLongFormat( lfmt );
   }

   ~TimeStamp() {}

   virtual void gcMark( uint32 mark ) {}

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
      Week starting on monday, 0 based. */
   int16 dayOfWeek() const;
   int64 toLongFormat() const;
   void fromLongFormat( int64 lf );
   void fromSystemTime( const SystemTime &st )
   {
      Sys::Time::timestampFromSystemTime( st, *this );
   }


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

   virtual TimeStamp *clone() const;

   //TODO: Add serialization
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
