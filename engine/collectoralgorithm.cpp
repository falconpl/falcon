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
#define RED_RETRY_TIMEOUT     250
#define YELLOW_RETRY_TIMEOUT  500
#define GREEN_RETRY_TIMEOUT   1000

#define MAX_TIMEOUT           10000

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
      coll->performGC(false);
      coll->memoryThreshold(memory + m_limit*2 );
      coll->algoTimeout(RED_RETRY_TIMEOUT);
   }
   else if( memory >= m_limit ) {
      coll->status( Collector::e_status_yellow );
      coll->suggestGC();
      coll->memoryThreshold(memory+m_limit);
      coll->algoTimeout(YELLOW_RETRY_TIMEOUT);
   }
   else {
      coll->status( Collector::e_status_green );
      coll->memoryThreshold(m_limit);
      coll->algoTimeout(0);
   }
}


bool CollectorAlgorithmFixed::onCheckComplete( Collector* coll )
{
   int64 amem = coll->activeMemory();
   int64 smem = coll->storedMemory();
   return (amem + m_limit) < smem;
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

CollectorAlgorithmRamp::CollectorAlgorithmRamp( int64 limit, numeric yellowFact, numeric redFact ):
      m_yellowLimit( (int64)(limit * yellowFact)  ),
      m_redLimit( (int64)(limit * redFact) ),
      m_yellowFactor( yellowFact ),
      m_redFactor( redFact )
{
   m_limit = limit;
   m_base = limit;
}


CollectorAlgorithmRamp::~CollectorAlgorithmRamp()
{}

void CollectorAlgorithmRamp::onApply( Collector* coll )
{
   coll->memoryThreshold( m_yellowLimit, true );
}

void CollectorAlgorithmRamp::onRemove( Collector* coll )
{
   // be sure to leave the timeout clean for who comes.
   coll->algoTimeout( 0 );
}


void CollectorAlgorithmRamp::onMemoryThreshold( Collector* coll, int64 threshold )
{
   if( threshold >= m_redLimit ) {
      coll->status( Collector::e_status_red );
      coll->suggestGC();
      coll->memoryThreshold( (int64)(m_redLimit*m_redFactor) );
      m_lastTimeout = ( (uint32)(RED_RETRY_TIMEOUT * m_yellowFactor) );
   }
   else if( threshold >= m_yellowLimit ) {
      coll->status( Collector::e_status_yellow );
      coll->suggestGC();
      coll->memoryThreshold( m_redLimit );
      m_lastTimeout = ( (uint32)(YELLOW_RETRY_TIMEOUT * m_yellowFactor) );
   }
   else
   {
      coll->memoryThreshold(m_yellowLimit);
      m_lastTimeout = ( (uint32)(GREEN_RETRY_TIMEOUT * m_yellowFactor) );
   }

   coll->algoTimeout( m_lastTimeout );
}


void CollectorAlgorithmRamp::onSweepComplete( Collector* coll, int64 freed, int64 )
{
   int64 memory = coll->storedMemory();
   // -- put current limit as next threshold
   m_mtx.lock();
   bool done = memory <= m_base;
   m_limit = done ? m_base : memory;
   int64 yl = m_yellowLimit = (int64)(m_limit * m_yellowFactor);
   m_redLimit = (int64)(m_limit * m_redFactor);
   m_mtx.unlock();

   // success in freeing?
   if( done )
   {
      // cancel pending timeout and set status
      coll->algoTimeout(0);
      coll->status(Collector::e_status_green);
   }
   else if( freed >= m_limit * (1-m_redFactor) )
   {
      // Rest the counter, we have hope in the future
      m_lastTimeout = TIMEOUT_TICK;
      coll->algoTimeout(m_lastTimeout);
   }
   // also add a threshold; whichever comes first wins...
   coll->memoryThreshold( yl );
}


bool CollectorAlgorithmRamp::onCheckComplete( Collector* coll )
{
   int64 amem = coll->activeMemory();
   int64 smem = coll->storedMemory();
   return (amem + m_redLimit) < smem;
}


void CollectorAlgorithmRamp::onTimeout( Collector* coll )
{
   int64 memory = coll->storedMemory();
   Collector::t_status status;
   bool suggestGC = false;
   //bool suggestMode = false;
   int64 yl = 0;

   m_mtx.lock();
   if( memory >= m_redLimit )
   {
     status = Collector::e_status_red;
     suggestGC = true;
     //suggestMode = true;
     m_lastTimeout += (uint32)(RED_RETRY_TIMEOUT * m_redFactor);
   }
   else if( memory >= m_yellowLimit ) {
     status = Collector::e_status_yellow;
     suggestGC = true;
     //suggestMode = false;
     m_lastTimeout += (uint32)(YELLOW_RETRY_TIMEOUT * m_redFactor);
   }
   else if( memory >= m_base )
   {
      // gets the limits a bit down and see what happens.
      m_limit =(int64)( m_limit/ m_yellowFactor);
      if( m_limit < m_base )
      {
         m_limit = m_base;
      }
      yl = m_yellowLimit = (int64)(m_limit * m_yellowFactor);
      m_redLimit = (int64)(m_limit * m_redFactor);

      if( memory >= m_redLimit )
      {
         status = Collector::e_status_red;
         m_lastTimeout += (uint32)(YELLOW_RETRY_TIMEOUT * m_redFactor);
      }
      else if( memory >= m_yellowLimit )
      {
         status = Collector::e_status_red;
         m_lastTimeout += (uint32)(YELLOW_RETRY_TIMEOUT * m_redFactor);
      }
      else {
         status = Collector::e_status_green;
         m_lastTimeout += (uint32)(GREEN_RETRY_TIMEOUT * m_redFactor);
      }
      // however, wait next time before suggesting collection

   }
   else {
     coll->memoryThreshold(m_yellowLimit);
     status = Collector::e_status_green;
     // timer off.
     m_lastTimeout = 0;
   }

   if( m_lastTimeout > MAX_TIMEOUT )
   {
      m_lastTimeout = MAX_TIMEOUT;
   }
   m_mtx.unlock();

   if( yl > 0 ) {
      coll->memoryThreshold( yl );
   }

   if( suggestGC )
   {
      coll->suggestGC( );
   }

   coll->status( status );
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
      ;
}


void CollectorAlgorithmRamp::limit( int64 l )
{
   m_mtx.lock();
   m_limit = l;
   if( m_limit < m_base )
   {
      m_limit = m_base;
   }
   m_yellowLimit = (int64)(m_limit * m_yellowFactor);
   m_redLimit = (int64)(m_limit * m_redFactor);
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
      m_yellowLimit = (int64)(m_limit * m_yellowFactor);
      m_redLimit = (int64)(m_limit * m_redFactor);
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
         CollectorAlgorithmRamp( 100000, 1.25, 1.5 )
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
         CollectorAlgorithmRamp( 1000000, 1.5, 2 )
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
         CollectorAlgorithmRamp( 10000000, 1.8, 2.5 )
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
