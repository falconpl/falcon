/*
   FALCON - The Falcon Programming Language.
   FILE: collectoralgorithm.h

   Ramp mode - progressive GC limits adjustment algorithms
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 18 Mar 2009 19:55:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_COLLECTOR_ALGORITHM_H
#define FALCON_COLLECTOR_ALGORITHM_H

#include <falcon/setup.h>
#include <falcon/types.h>

#include <stdlib.h>
#include <falcon/collector.h>

namespace Falcon {

/** Ramp-up GC parameters base class.
   The subclasses of this virtual class contain a configurable algorithm used by the
   Memory Pool to update its status after a successful garbage collection.
*/
class FALCON_DYN_CLASS CollectorAlgorithm
{
public:
   CollectorAlgorithm() {}
   virtual ~CollectorAlgorithm() {};

   virtual Collector::t_status checkStatus() = 0;
   virtual void onSweepComplete( int64 allocatedMemory, int64 allocatedItems ) = 0;
   virtual void describe( String& target) const = 0;

   String describe() const {
      String temp; describe( temp ); return temp;
   }
};


/** This collector algorithm disables automatic GC.
   The GC never punches in.
*/
class FALCON_DYN_CLASS CollectorAlgorithmNone: public CollectorAlgorithm
{
public:
   CollectorAlgorithmNone():
      CollectorAlgorithm()
   {}
   virtual ~CollectorAlgorithmNone() {};
   virtual Collector::t_status checkStatus() { return Collector::e_status_green; }
   virtual void onSweepComplete( int64, int64 ) {}

   virtual void describe( String& target) const { target = "CollectorAlgorithmNone"; }
};


/** This algorithm sets a three fixed levels at which GC is started.

*/
class FALCON_DYN_CLASS CollectorAlgorithmFixed: public CollectorAlgorithm
{
public:
   CollectorAlgorithmFixed( int64 thresholdYellow, int64 thresholdBrown, int64 thresholdRed );
   virtual ~CollectorAlgorithmFixed() {};

   virtual Collector::t_status checkStatus();
   virtual void onSweepComplete( int64 allocatedMemory, int64 allocatedItems );
   virtual void describe(String& target) const;

   void setLevels( int64 yellow, int64 brown, int64 red );
   void setYellow( int64 value );
   void setBrown( int64 value );
   void setRed( int64 value );

   void getLevels( int64& yellow, int64& brown, int64& red ) const;

protected:
   int64 m_thresholdYellow;
   int64 m_thresholdBrown;
   int64 m_thresholdRed;

   mutable Mutex m_mtx;
};

/** Base class for all ramping algorithms. */
class FALCON_DYN_CLASS CollectorAlgorithmRamp: public CollectorAlgorithmFixed
{
public:
   CollectorAlgorithmRamp();
   virtual ~CollectorAlgorithmRamp();

   virtual Collector::t_status checkStatus();
   virtual void onSweepComplete( int64 allocatedMemory, int64 allocatedItems );
   virtual void describe( String& target) const;

protected:
   Collector::t_status m_currentStatus;
   int64 m_sweptMemory;

   numeric m_yellowBaseRatio;
   numeric m_brownBaseRatio;
   numeric m_redBaseRatio;

   numeric m_yellowMaxRatio;
   numeric m_brownMaxRatio;
   numeric m_redMaxRatio;

   numeric m_growthRatio;
   numeric m_successRatio;

   int32 m_sweepThreshold;
   int32 m_sweptTimes;
};

/** Enforces a strict inspection policy.
  Starts at red;
  A successful sweep is defined as a sweep reducing the used memory by 10%.

  After a successful sweep:
  - yellow is half the live memory.
  - brown is 75% live memory
  - red is the live memory.

  After each non-successful sweep, levels are raised 5% up to when
  - yellow is the live memory  +  1
  - brown is the live memory * 1.5
  - red is the live memory * 2

  After 50 sweeps, a successful sweep is implied and counters are reset down.
*/
class FALCON_DYN_CLASS CollectorAlgorithmStrict: public CollectorAlgorithmRamp
{
public:
   CollectorAlgorithmStrict();
   virtual ~CollectorAlgorithmStrict();

   void describe( String& target ) const;
};


/** Enforces a smooth inspection policy.
  Starts at brown;
  A successful sweep is defined as a sweep reducing the used memory by 15%.

  After a successful sweep:
  - yellow is 75% the live memory.
  - brown is 125% live memory
  - red is 150% live memory.

  After each non-successful sweep, levels are raised 10% up to when
  - yellow is 100% the live memory
  - brown is 150% the live memory
  - red is 200% the live memory
*/
class FALCON_DYN_CLASS CollectorAlgorithmSmooth: public CollectorAlgorithmRamp
{
public:
   CollectorAlgorithmSmooth();
   virtual ~CollectorAlgorithmSmooth();

   void describe( String& target ) const;
};


/** Enforces a loose inspection policy.
  Starts at yellow;
  A successful sweep is defined as a sweep reducing the used memory by 25%.

  After a successful sweep:
  - yellow is 100% the live memory.
  - brown is 150% live memory
  - red is 200% live memory.

  After a non-succesful sweep, all levels are raised by 10% up to when
  - yellow is 75% the live memory.
  - brown is 125% live memory
  - red is 150% live memory.
*/
class FALCON_DYN_CLASS CollectorAlgorithmLoose: public CollectorAlgorithmRamp
{
public:
   CollectorAlgorithmLoose();
   virtual ~CollectorAlgorithmLoose();
   void describe( String& target ) const;
};



#define FALCON_COLLECTOR_ALGORITHM_OFF       0
#define FALCON_COLLECTOR_ALGORITHM_FIXED     1
#define FALCON_COLLECTOR_ALGORITHM_STRICT    2
#define FALCON_COLLECTOR_ALGORITHM_SMOOTH    3
#define FALCON_COLLECTOR_ALGORITHM_LOOSE     4
#define FALCON_COLLECTOR_ALGORITHM_COUNT     5

#define FALCON_COLLECTOR_ALGORITHM_DEFAULT   3


}

#endif

/* end of collectoralgorithm.h */
