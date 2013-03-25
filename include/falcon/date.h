/*
   FALCON - The Falcon Programming Language.
   FILE: date.h

   Utility to create embedding applications.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 23 Mar 2013 16:05:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DATE_H_
#define _FALCON_DATE_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/sys.h>

#include <math.h>

namespace Falcon {

class DataWriter;
class DataReader;

/**
 * Abstract representation of a universal moment in time.
 *
 * The precision is up to femtoseconds since the big-bang.
 *
 * The data is opaque, and can be converted through utility
 * functions or class constructors. In particular, TimeStamp
 * has support to convert this entity into a gregorian calendar date.
 *
 */
class Date
{
public:
   static const int64 MAX_FEMTOSECOND      =  999999999999999LL;
   static const int64 FEMTOSECONDS         = 1000000000000000LL;
   static const int64 MILLISECOND_DIVIDER  = 1000000000000LL;
   static const int64 MICROSECOND_DIVIDER  = 1000000000LL;
   static const int64 NANOSECOND_DIVIDER   = 1000000LL;
   static const int64 PICOSECOND_DIVIDER   = 1000LL;

   /** Creates an empty date.
    * The date is set at epoch (1/1/1970 00 GMT)
    */
   Date():
      m_seconds(0),
      m_femtoseconds(0)
   {}

   /**
    * Creates a date at an arbitrary moment in time.
    * @param seconds Seconds relative from epoch (1/1/1970 00 GMT).
    * @param fs Femtoseconds, 0 to 1e15 -1 (fifteen nines in a row)
    */
   Date( int64 seconds, int64 fs ):
      m_seconds(seconds),
      m_femtoseconds(fs)
   {}

   /**
    * Copies another date.
    */
   Date( const Date& other ):
      m_seconds(other.m_seconds),
      m_femtoseconds(other.m_femtoseconds)
   {}

   ~Date() {}

   /** Changes the date into a current date. */
   Date& setCurrent()
   {
      Sys::_getCurrentDate(*this);
      return *this;
   }

   /**
    * Gets the current system time as a date.
    *
    */
   inline static Date current() {
      Date dt;
      return dt.setCurrent();
   }

   /**
    * Gets the seconds since epoch.
    */
   int64 seconds() const { return m_seconds; }
   /**
    * Sets the seconds since epoch.
    */
   void seconds(int64 s ) { m_seconds = s; }

   /**
    * Gets the femtoseconds (1e-15 secs) since the beginning of the current second in the date.
    */
   int64 femtoseconds() const { return m_femtoseconds; }

   /**
    * Sets the femtoseconds (1e-15 secs) since the beginning of the current second in the date.
    */
   void femtoseconds(int64 s) { m_femtoseconds = s; }

   Date& operator=(const Date& d)
   {
      m_seconds = d.m_seconds;
      m_femtoseconds = d.m_femtoseconds;
      return *this;
   }

   bool operator==(const Date& d) const
   {
      return m_seconds == d.m_seconds && m_femtoseconds == d.m_femtoseconds;
   }

   bool operator!=(const Date& d) const
   {
      return m_seconds != d.m_seconds || m_femtoseconds != d.m_femtoseconds;
   }

   bool operator<(const Date& d) const
   {
      return m_seconds < d.m_seconds || ( m_seconds == d.m_seconds && m_femtoseconds < d.m_femtoseconds);
   }

   bool operator<=(const Date& d) const
   {
      return m_seconds < d.m_seconds || ( m_seconds == d.m_seconds && m_femtoseconds <= d.m_femtoseconds);
   }

   bool operator>(const Date& d) const
   {
      return m_seconds > d.m_seconds || ( m_seconds == d.m_seconds && m_femtoseconds > d.m_femtoseconds);
   }

   bool operator>=(const Date& d) const
   {
      return m_seconds > d.m_seconds || ( m_seconds == d.m_seconds && m_femtoseconds >= d.m_femtoseconds);
   }

   Date operator -() const
   {
      return Date( -m_seconds, -m_femtoseconds );
   }

   Date& operator +=( const Date& other )
   {
      m_seconds += other.m_seconds;

      // same sign, the fractional part is just summed up and rounded to max
      m_femtoseconds += other.m_femtoseconds;

      roundFemto();
      return *this;
   }

   Date& operator -=( const Date& other )
   {
      m_seconds -= other.m_seconds;

      // same sign, the fractional part is just summed up and rounded to max
      m_femtoseconds -= other.m_femtoseconds;

      roundFemto();
      return *this;
   }


   numeric toSeconds() const
   {
      numeric res = m_seconds * 1000.0;
      res += m_femtoseconds / 1e12;
      return res;
   }


   void fromSeconds( numeric ts )
   {
      m_seconds = ts / 1000.0;
      numeric intpart = 0.0;
      m_femtoseconds = modf(ts, &intpart);
      m_femtoseconds *= 1e12;
   }


   int64 toMilliseconds() const
   {
      int64 res = m_seconds * 1000LL;
      res += (int64) m_femtoseconds / 1000000000000LL;
      return res;
   }


   /**
    * Sets the date from a value that express milliseconds since epoch.
    */
   void fromMilliseconds( int64 msecs )
   {
      m_seconds = msecs / 1000LL;
      m_femtoseconds = msecs % 1000;
      m_femtoseconds *= MILLISECOND_DIVIDER;
   }

   void addSeconds( int64 secs )
   {
      m_seconds += secs;
   }

   void addMilliseconds( int64 msecs )
   {
      m_seconds += msecs/1000;
      m_femtoseconds += static_cast<int64>(msecs % 1000) * 1000000000000LL;
      roundFemto();
   }

private:
   int64 m_seconds;
   int64 m_femtoseconds;

   friend class DataWriter;
   friend class DataReader;

   void roundFemto()
   {
      //if les than -10...00, round down
      if( m_femtoseconds <= -FEMTOSECONDS)
      {
         // remember, seconds and femtoseconds have the same sign.
         m_seconds --;
         m_femtoseconds += FEMTOSECONDS;
      }
      else if( m_femtoseconds >= FEMTOSECONDS )
      {
         m_seconds ++;
         m_femtoseconds -= FEMTOSECONDS;
      }
      else if ( m_seconds > 0 && m_femtoseconds < 0 )
      {
         m_seconds--;
         m_femtoseconds += FEMTOSECONDS;
      }
      else if( m_seconds < 0 && m_femtoseconds > 0 )
      {
         m_seconds++;
         m_femtoseconds -= FEMTOSECONDS;
      }
   }
};


inline Date operator+(const Date& date1, const Date& date2 )
{
   Date dr( date1 );
   dr += date2;
   return dr;
}


inline Date operator-(const Date& date1, const Date& date2 )
{
   Date dr( date1 );
   dr -= date2;
   return dr;
}

}

#endif

/* end of date.h */
