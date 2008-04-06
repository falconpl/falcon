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
#include <falcon/userdata.h>
#include <falcon/time_sys.h>

namespace Falcon {

class String;

/** TimeStamp class.
   This class is both used as a system independent time accounting
   object and as a internal object for the TimeStamp falcon core
   object.
*/
class FALCON_DYN_CLASS TimeStamp: public UserData
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

   void copy( const TimeStamp &ts );
   void currentTime();
   bool isValid() const;
   bool isLeapYear() const;
   int16 dayOfYear() const;
   int16 dayOfWeek() const;
   int64 toLongFormat() const;
   void fromLongFormat( int64 lf );
   void fromSystemTime( const SystemTime &st )
   {
      Sys::Time::timestampFromSystemTime( st, *this );
   }

   numeric asSeconds() const;
   void fromSeconds( numeric seconds );

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

   friend void Sys::Time::timestampFromSystemTime( const SystemTime &sys_time, TimeStamp &ts );

   virtual bool isReflective() const;
   virtual void getProperty( VMachine *, const String &propName, Item &prop );
   virtual void setProperty( VMachine *, const String &propName, Item &prop );

   virtual UserData *clone() const;
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
