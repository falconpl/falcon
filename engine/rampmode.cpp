/*
   FALCON - The Falcon Programming Language.
   FILE: rampmode.cpp

   Ramp mode - progressive GC limits adjustment algoritmhs
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 19 Mar 2009 08:23:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/rampmode.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>

namespace Falcon {

RampMode::~RampMode()
{}

void RampMode::reset()
{
}

//=========================================================
//


RampStrict::~RampStrict()
{
}

void RampStrict::onScanInit()
{
}

void RampStrict::onScanComplete()
{
   m_active = gcMemAllocated();
   m_normal = m_active/2;
}


//=========================================================
//


RampLoose::~RampLoose()
{
}

void RampLoose::onScanInit()
{
   m_active = (size_t)(gcMemAllocated() );
   m_normal = (size_t)(gcMemAllocated() / 2 );
}

void RampLoose::onScanComplete()
{
}

//=========================================================
//

RampSmooth::RampSmooth( numeric factor ):
   RampMode(),
   m_pNormal(0),
   m_pActive(0),
   m_factor( factor )
{
   if ( m_factor < 1.0 )
      m_factor = 1.0;
}


RampSmooth::~RampSmooth()
{
}

void RampSmooth::reset()
{
   m_pNormal = 0;
}


void RampSmooth::onScanInit()
{
   // on the first loop, we setup the waiting loops.
   if ( m_pNormal == 0 )
   {
      m_pNormal = gcMemAllocated();
   }
   else {
      // size_t is usually unsigned.
      size_t allocated = gcMemAllocated();
      if( m_pNormal > allocated )
      {
         // we're getting smaller
         m_pNormal -= (size_t)((m_pNormal - allocated) / m_factor);
      }
      else {
         m_pNormal += size_t((allocated-m_pNormal) / m_factor);
      }
   }

   m_normal = m_pNormal;
   m_active = (size_t)(m_normal * m_factor);
}

void RampSmooth::onScanComplete()
{
}

}

/* end of rampmode.cpp */
