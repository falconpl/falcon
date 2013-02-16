/*
   FALCON - The Falcon Programming Language.
   FILE: collectoralgorithm.cpp

   Ramp mode - progressive GC limits adjustment algorithms
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 19 Mar 2009 08:23:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "/engine/collectoralgorithm.cpp"

#include <falcon/collectoralgorithm.h>
#include <falcon/collector.h>
#include <falcon/engine.h>

#include <falcon/log.h>

#define TIMEOUT_TICK          200
#define RED_RETRY_TIMEOUT     500
#define YELLOW_RETRY_TIMEOUT  1000
#define GREEN_RETRY_TIMEOUT   2000

namespace Falcon {

//=======================================================================
// Fixed algorithm.
//

CollectorAlgorithmFixed::CollectorAlgorithmFixed( int64 limit ):
         CollectorAlgorithm(),
         m_limit(limit)
{}

void CollectorAlgorithmFixed::onApply(Collector* coll)
{
   coll->memoryThreshold(m_limit);
}


void CollectorAlgorithmFixed::onMemoryThreshold( Collector* coll, int64 memory )
{
   if( memory >= m_limit * 2 )
   {
      coll->status( Collector::e_status_red );
      coll->suggestGC(true);
      coll->memoryThreshold(memory + m_limit*2 );
      coll->algoTimeout(RED_RETRY_TIMEOUT);
   }
   else if( memory >= m_limit ) {
      coll->status( Collector::e_status_yellow );
      coll->suggestGC(false);
      coll->memoryThreshold(memory+m_limit);
      coll->algoTimeout(YELLOW_RETRY_TIMEOUT);
   }
   else {
      coll->status( Collector::e_status_green );
      coll->memoryThreshold(m_limit);
      coll->algoTimeout(0);
   }
}


void CollectorAlgorithmFixed::onTimeout( Collector* coll )
{
   onMemoryThreshold(coll, coll->storedMemory() );
   // let sweep to reschedule us if necessary
   // we might be called by the sweep loop; see if we should
}


void CollectorAlgorithmFixed::limit( int64 l )
{
   m_limit = l;
}

int64 CollectorAlgorithmFixed::limit() const
{
   return m_limit;
}

void CollectorAlgorithmFixed::base( int64  )
{
}

int64 CollectorAlgorithmFixed::base() const
{
   return 0;
}


void CollectorAlgorithmFixed::onSweepComplete( Collector* coll, int64 freedMem, int64 freedItems )
{
   static Log* log = Engine::instance()->log();

   int64 mem = coll->storedMemory();
   if ( mem < m_limit )
   {
      coll->status( Collector::e_status_green );
      coll->memoryThreshold(m_limit );
      // cancel pending timeouts.
      coll->algoTimeout(0);
   }
   else if ( freedItems > 0 )
   {
      coll->status(Collector::e_status_yellow);
      coll->algoTimeout(YELLOW_RETRY_TIMEOUT);
   }
   else {
      coll->status(Collector::e_status_red);
      coll->algoTimeout(RED_RETRY_TIMEOUT);
   }

   log->log(Log::fac_engine, Log::lvl_detail, String("Swept ").N(freedMem) );
}


void CollectorAlgorithmFixed::describe(String& target) const
{
   target = "CollectorAlgorithmFixed(";
   target.N(m_limit).A(")");
}


//=======================================================================
// Ramp algorithm.
//

CollectorAlgorithmRamp::CollectorAlgorithmRamp( int64 limit, numeric sweepFact, numeric yellowFact, numeric redFact ):
      m_sweepThreshold( limit * sweepFact ),
      m_yellowLimit( limit * yellowFact  ),
      m_redLimit( limit * redFact ),
      m_yellowFactor( yellowFact ),
      m_redFactor( redFact ),
      m_sweepFactor( sweepFact )
{
   m_limit = limit;
   m_base = limit * (1-sweepFact);
}


CollectorAlgorithmRamp::~CollectorAlgorithmRamp()
{}

void CollectorAlgorithmRamp::onApply( Collector* coll )
{
   coll->memoryThreshold( m_yellowLimit );
}

void CollectorAlgorithmRamp::onRemove( Collector* coll )
{
   // be sure to leave the timeout clean for who comes.
   coll->algoTimeout( 0 );
}


void CollectorAlgorithmRamp::onMemoryThreshold( Collector* coll, int64 threshold )
{
   if( threshold >= m_redLimit *m_redFactor ) {
      coll->status( Collector::e_status_red );
      coll->suggestGC(true);
   }
   if( threshold >= m_redLimit ) {
      coll->status( Collector::e_status_red );
      coll->memoryThreshold( m_redLimit* m_redFactor );
      coll->suggestGC(false);
   }
   else if( threshold >= m_yellowLimit ) {
      coll->status( Collector::e_status_yellow );
      coll->memoryThreshold( m_redLimit );
   }
   else {
      coll->memoryThreshold(m_yellowLimit);
   }
}


void CollectorAlgorithmRamp::onSweepComplete( Collector* coll, int64 freedMemory, int64 )
{
   Collector::t_status status = coll->status();

   int64 memory = coll->storedMemory();
   // -- put current limit as next threshold
   m_mtx.lock();
   int64 oldLimit = m_limit;
   m_limit = memory > m_base ? memory : m_base;
   int64 yl = (m_yellowLimit = m_limit * m_yellowFactor);
   int64 rl = (m_redLimit = m_limit * m_redFactor);
   int64 oldThreshold = m_sweepThreshold;
   m_sweepThreshold = m_limit * m_sweepFactor;
   m_mtx.unlock();

   // success in freeing memory?
   if( freedMemory >= oldThreshold || memory < oldLimit )
   {
      // -- go down a state.
      if ( status == Collector::e_status_red )
      {
         coll->status(Collector::e_status_yellow);
         yl = rl;
      }
      else if ( status == Collector::e_status_yellow ) {
         coll->status(Collector::e_status_green);
         // keep current to
      }
      m_lastTimeout = TIMEOUT_TICK * (10*m_redFactor);
   }
   else {
      m_lastTimeout = TIMEOUT_TICK*(5*m_redFactor);
   }

   coll->algoTimeout(m_lastTimeout);
   coll->memoryThreshold( yl, true );
}


void CollectorAlgorithmRamp::onTimeout( Collector* coll )
{
   static Log* log = Engine::instance()->log();
   Collector::t_status status = coll->status();

   log->log(Log::fac_engine, Log::lvl_detail,
               String("Timeout for collector strategy ").N(status) );

   // if we're still in yellow status or above, try to suggest some collection.
   if ( status == Collector::e_status_red &&  (m_redLimit*2 < coll->storedMemory()) )
   {
      coll->suggestGC(true);
      if( m_lastTimeout > TIMEOUT_TICK*2 )
      {
         // shrink the timeout
         m_lastTimeout /= m_redFactor;
      }
   }
   else if ( status == Collector::e_status_red )
   {
      coll->suggestGC(false);
      // keep current timeout
   }
   else if ( status == Collector::e_status_yellow ) {
      if( m_lastTimeout < TIMEOUT_TICK*25 )
      {
         // grow the timeout
         m_lastTimeout *= m_yellowFactor;
      }

      coll->suggestGC(false);
   }
   else {
      if( m_lastTimeout < TIMEOUT_TICK*50 )
      {
         // grow the timeout more
         m_lastTimeout *= m_redFactor;
      }

      // suggest some gc.
      coll->status( Collector::e_status_yellow );
      coll->suggestGC(false);
   }

   coll->algoTimeout( m_lastTimeout );
}


void CollectorAlgorithmRamp::describe( String& target ) const
{
   // we don't care to lock here.
   target
      .A("ylmt:").N(m_yellowLimit)
      .A(", rlmt:").N(m_redLimit)
      .A(", yfac:").N(m_yellowFactor)
      .A(", rfac:").N(m_redFactor)
      .A(", swt:").N(m_sweepThreshold);
}


void CollectorAlgorithmRamp::limit( int64 l )
{
   m_mtx.lock();
   m_limit = l;
   if( m_limit < m_base )
   {
      m_limit = m_base;
   }
   m_yellowLimit = m_limit * m_yellowFactor;
   m_redLimit = m_limit * m_redFactor;
   m_mtx.unlock();
}

int64 CollectorAlgorithmRamp::limit() const
{
   m_mtx.lock();
   int64 l = m_limit;
   m_mtx.unlock();

   return l;
}

void CollectorAlgorithmRamp::base( int64 l )
{
   m_mtx.lock();
   m_base = l;
   if( m_limit < m_base )
   {
      m_limit = m_base;
      m_yellowLimit = m_limit * m_yellowFactor;
      m_redLimit = m_limit * m_redFactor;
   }
   m_mtx.unlock();
}

int64 CollectorAlgorithmRamp::base() const
{
   m_mtx.lock();
   int64 l = m_base;
   m_mtx.unlock();

   return l;
}


//=======================================================================
// Strict algorithm.
//


CollectorAlgorithmStrict::CollectorAlgorithmStrict():
         CollectorAlgorithmRamp( 100000, 0.5, 1.25, 1.5 )
{
}

CollectorAlgorithmStrict::~CollectorAlgorithmStrict()
{
}

void CollectorAlgorithmStrict::describe( String& target ) const
{
   target = "CollectorAlgorithmStrict ";
   CollectorAlgorithmRamp::describe(target);
}


//=======================================================================
// Smooth Algorithm.
//

CollectorAlgorithmSmooth::CollectorAlgorithmSmooth():
         CollectorAlgorithmRamp( 1000000, 0.25, 1.5, 2 )
{
}

CollectorAlgorithmSmooth::~CollectorAlgorithmSmooth()
{
}

void CollectorAlgorithmSmooth::describe( String& target ) const
{
   target = "CollectorAlgorithmSmooth ";
   CollectorAlgorithmRamp::describe(target);
}


//=======================================================================
// Smooth Algorithm.
//

CollectorAlgorithmLoose::CollectorAlgorithmLoose():
         CollectorAlgorithmRamp( 10000000, 0.1, 2, 3 )
{
}

CollectorAlgorithmLoose::~CollectorAlgorithmLoose()
{
}

void CollectorAlgorithmLoose::describe( String& target ) const
{
   target = "CollectorAlgorithmLoose ";
   CollectorAlgorithmRamp::describe(target);
}


}

/* end of collectoralgorithm.cpp */
