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

/** Base class for collection algorithms.
 *
 * An instance of this class can be set in the GC to be
 * notified about relevant events happening in the garbage
 * collector.
 *
 * The subclasses should set the GC status, invoke Collector::suggestGC
 * or Collector::performGC on their perception of the status
 * of the memory.
 *
 * The algorithm could install a timer callback that is periodically
 * called in the GC, or a memory / item
 * threshold callback that will be invoked by the GC when the limits
 * are broken.
 *
 * The class has also hooks called before and after sweep loops, so that
 * it is possible to determine the efficacy of a just started loop.
 *
 * \note callbacks are called from different threads. If necessary,
 * use locking or atomic access to your variables.
 *
*/
class FALCON_DYN_CLASS CollectorAlgorithm
{
public:
   CollectorAlgorithm() {}
   virtual ~CollectorAlgorithm() {};

   virtual void onApply(Collector* coll) = 0;
   virtual void onRemove(Collector* coll) = 0;

   virtual bool onCheckComplete( Collector* coll ) = 0;

   virtual void onMemoryThreshold( Collector* coll, int64 threshold ) = 0;
   virtual void onItemThreshold( Collector* coll, int64 threshold ) = 0;

   virtual void onSweepBegin( Collector* coll ) = 0;
   virtual void onSweepComplete( Collector* coll, int64 freedMemry, int64 freedItems ) = 0;

   virtual void describe(String& target) const = 0;

   virtual void onTimeout( Collector* ) = 0;

   virtual void limit( int64 l ) = 0;
   virtual int64 limit() const = 0;
   virtual void base( int64 l ) = 0;
   virtual int64 base() const = 0;

   String describe() const {
      String temp; describe( temp ); return temp;
   }
};


class FALCON_DYN_CLASS CollectorAlgorithmManual: public CollectorAlgorithm
{
public:
   CollectorAlgorithmManual() {}
   virtual ~CollectorAlgorithmManual() {}

   virtual void onApply(Collector* ) {}
   virtual void onRemove(Collector* ) {}

   virtual void onMemoryThreshold( Collector* , int64 ) {}
   virtual void onItemThreshold( Collector* , int64 ) {}
   virtual bool onCheckComplete( Collector* ) { return false; }

   virtual void onSweepBegin( Collector* ) {}
   virtual void onSweepComplete( Collector*, int64, int64 ) {}

   virtual void describe(String& target) const { target = "Manual";}

   virtual void onTimeout( Collector* ) {}

   void limit( int64 ) {}
   int64 limit() const { return 0; }
   void base( int64 ) {}
   int64 base() const { return 0; }

   String describe() const {
      String temp; describe( temp ); return temp;
   }
};


/**

*/
class FALCON_DYN_CLASS CollectorAlgorithmFixed: public CollectorAlgorithm
{
public:
   CollectorAlgorithmFixed( int64 limit );
   virtual ~CollectorAlgorithmFixed() {}

   virtual void onApply(Collector* coll);
   virtual void onRemove(Collector* ) {}

   virtual void onMemoryThreshold( Collector* coll, int64 threshold );
   virtual void onItemThreshold( Collector* , int64  ) {}
   virtual bool onCheckComplete( Collector* coll );

   virtual void onSweepBegin( Collector* ) {}
   virtual void onSweepComplete( Collector* coll, int64 freedMemory, int64 freedItems );
   virtual void onTimeout( Collector* );

   virtual void describe(String& target) const;

   void setLimit( int64 limit ) { m_limit = limit; }

   int64 getLimit() const { return m_limit; }

   void limit( int64 );
   int64 limit() const;
   void base( int64 );
   int64 base() const;

protected:
   int64 m_limit;
};

/** Base class for all ramping algorithms. */
class FALCON_DYN_CLASS CollectorAlgorithmRamp: public CollectorAlgorithm
{
public:
   CollectorAlgorithmRamp( int64 limit, numeric yellowFact, numeric redFact );
   virtual ~CollectorAlgorithmRamp();

   virtual void onApply(Collector* );
   virtual void onRemove(Collector* );

   virtual void onMemoryThreshold( Collector* coll, int64 threshold );
   virtual void onItemThreshold( Collector* , int64  ) {}
   virtual bool onCheckComplete( Collector* coll );

   virtual void onSweepBegin( Collector* ) {}
   virtual void onSweepComplete( Collector* coll, int64 freedMemory, int64 freedItems );
   virtual void onTimeout( Collector* );


   void limit( int64 );
   int64 limit() const;
   void base( int64 );
   int64 base() const;

   virtual void describe( String& target) const;

protected:
   int64 m_base;
   int64 m_limit;

   int64 m_yellowLimit;
   int64 m_redLimit;
   numeric m_yellowFactor;
   numeric m_redFactor;

   uint32 m_lastTimeout;
   mutable Mutex m_mtx;
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



#define FALCON_COLLECTOR_ALGORITHM_MANUAL    0
#define FALCON_COLLECTOR_ALGORITHM_FIXED     1
#define FALCON_COLLECTOR_ALGORITHM_STRICT    2
#define FALCON_COLLECTOR_ALGORITHM_SMOOTH    3
#define FALCON_COLLECTOR_ALGORITHM_LOOSE     4
#define FALCON_COLLECTOR_ALGORITHM_COUNT     5

#define FALCON_COLLECTOR_ALGORITHM_DEFAULT   3


}

#endif

/* end of collectoralgorithm.h */
