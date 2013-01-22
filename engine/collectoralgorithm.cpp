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

namespace Falcon {

//=======================================================================
// Fixed algorithm.
//

CollectorAlgorithmFixed::CollectorAlgorithmFixed( int64 yellow, int64 brown, int64 red ):
         CollectorAlgorithm(),
         m_thresholdYellow(yellow),
         m_thresholdBrown(brown),
         m_thresholdRed(red)
{}


Collector::t_status CollectorAlgorithmFixed::checkStatus()
{
   int64 level = Engine::collector()->storedMemory();

   Collector::t_status result;
   m_mtx.lock();
   if( level <= m_thresholdYellow ) {
      result = Collector::e_status_green;
   }
   else if( level <= m_thresholdBrown ) {
      result = Collector::e_status_yellow;
   }
   else if( level <= m_thresholdRed ) {
      result = Collector::e_status_brown;
   }
   else {
      result = Collector::e_status_red;
   }
   m_mtx.unlock();

   return result;
}


void CollectorAlgorithmFixed::onSweepComplete( int64, int64 )
{
}

void CollectorAlgorithmFixed::describe(String& target) const
{
   target = "CollectorAlgorithmFixed(";
   target.N(m_thresholdYellow).A(", ").N(m_thresholdBrown).A(", ").N(m_thresholdRed);
}

void CollectorAlgorithmFixed::setLevels( int64 yellow, int64 brown, int64 red )
{
   m_mtx.lock();
   m_thresholdYellow = yellow;
   m_thresholdBrown = brown;
   m_thresholdRed = red;
   m_mtx.unlock();
}
void CollectorAlgorithmFixed::setYellow( int64 value )
{
   m_mtx.lock();
   m_thresholdYellow = value;
   m_mtx.unlock();
}

void CollectorAlgorithmFixed::setBrown( int64 value )
{
   m_mtx.lock();
   m_thresholdBrown = value;
   m_mtx.unlock();
}

void CollectorAlgorithmFixed::setRed( int64 value )
{
   m_mtx.lock();
   m_thresholdRed = value;
   m_mtx.unlock();
}

void CollectorAlgorithmFixed::getLevels( int64& yellow, int64& brown, int64& red ) const
{
   m_mtx.lock();
   yellow = m_thresholdYellow;
   brown = m_thresholdBrown;
   red = m_thresholdRed;
   m_mtx.unlock();
}


//=======================================================================
// Ramp algorithm.
//

CollectorAlgorithmRamp::CollectorAlgorithmRamp():
   CollectorAlgorithmFixed(0,0,0)
{
   m_sweptMemory = 0;
   m_sweptTimes = 0;
}

CollectorAlgorithmRamp::~CollectorAlgorithmRamp()
{}

Collector::t_status CollectorAlgorithmRamp::checkStatus()
{
   m_mtx.lock();
   Collector::t_status result = m_currentStatus;
   m_mtx.unlock();

   return result;
}

void CollectorAlgorithmRamp::onSweepComplete( int64 allocatedMemory, int64 )
{
   m_mtx.lock();
   if( allocatedMemory <= m_thresholdYellow ) {
      m_currentStatus = Collector::e_status_green;
   }
   else if( allocatedMemory <= m_thresholdBrown ) {
      m_currentStatus = Collector::e_status_yellow;
   }
   else if( allocatedMemory <= m_thresholdRed ) {
      m_currentStatus = Collector::e_status_brown;
   }
   else {
      m_currentStatus = Collector::e_status_red;
   }

   if( m_sweptMemory == 0 || m_sweptMemory >= allocatedMemory * (1+m_successRatio) )
   {
      // a successful event.
      m_thresholdYellow = allocatedMemory * m_yellowBaseRatio;
      m_thresholdBrown = allocatedMemory * m_brownBaseRatio;
      m_thresholdRed = allocatedMemory * m_redBaseRatio;
   }
   else {
      if( m_thresholdYellow *(1+m_growthRatio) <= m_sweptMemory * m_yellowMaxRatio )
      {
         m_thresholdYellow *= (1+m_growthRatio);
      }

      if( m_thresholdBrown *(1+m_growthRatio) < m_sweptMemory * m_brownMaxRatio )
      {
         m_thresholdBrown *= (1+m_growthRatio);
      }

      if( m_thresholdRed *(1+m_growthRatio) < m_sweptMemory * m_redMaxRatio )
      {
         m_thresholdRed *= (1+m_growthRatio);
      }
   }

   if( m_sweepThreshold > 0 && m_sweptTimes++ >= m_sweepThreshold ) {
      m_sweptMemory = 0;
   }
   else {
      m_sweptMemory = allocatedMemory;
   }

   m_mtx.unlock();
}

void CollectorAlgorithmRamp::describe( String& target) const
{
   switch( m_currentStatus ) {
   case Collector::e_status_green: target.A("green "); break;
   case Collector::e_status_yellow: target.A("yellow "); break;
   case Collector::e_status_brown: target.A("brown "); break;
   case Collector::e_status_red: target.A("red "); break;
   default: target.A("forced "); break;
   }

   target.N(m_thresholdYellow).A(", ").N(m_thresholdBrown).A(", ").N(m_thresholdRed)
            .A( "[Sw: " ).N(m_sweptMemory).A(" sr:").N(m_successRatio).A("]{")
            .N(m_sweptTimes).A("/").N(m_sweepThreshold).A("}");
}

//=======================================================================
// Strict algorithm.
//


CollectorAlgorithmStrict::CollectorAlgorithmStrict():
         CollectorAlgorithmRamp()
{
   m_currentStatus = Collector::e_status_red;

   m_yellowBaseRatio = 0.5;
   m_brownBaseRatio = 0.75;
   m_redBaseRatio = 1.0001;

   m_yellowMaxRatio = 0.98;
   m_brownMaxRatio = 1.25;
   m_redMaxRatio = 1.50;

   m_growthRatio = 0.05;
   m_successRatio = 0.05;

   m_sweepThreshold = 10;
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
         CollectorAlgorithmRamp()
{
   m_currentStatus = Collector::e_status_brown;

   m_yellowBaseRatio = 0.75;
   m_brownBaseRatio = 1.0001;
   m_redBaseRatio = 1.50;

   m_yellowMaxRatio = 1.001;
   m_brownMaxRatio = 1.50;
   m_redMaxRatio = 2.0;

   m_growthRatio = 0.15;
   m_successRatio = 0.15;

   m_sweepThreshold = 50;
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
         CollectorAlgorithmRamp()
{
   m_currentStatus = Collector::e_status_yellow;

   m_yellowBaseRatio = 0.75;
   m_brownBaseRatio = 1.0001;
   m_redBaseRatio = 1.50;

   m_yellowMaxRatio = 1.001;
   m_brownMaxRatio = 1.50;
   m_redMaxRatio = 2.0;

   m_growthRatio = 0.25;
   m_successRatio = 0.25;

   m_sweepThreshold = 0;
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
