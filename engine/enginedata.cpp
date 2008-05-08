/*
   FALCON - The Falcon Programming Language
   FILE: enginedata.cpp

   Engine static data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun dic 4 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Engine static data setup and initialization
*/

#include <falcon/enginedata.h>

namespace Falcon {

namespace Engine {

// Default (nil) functions
static volatile long s_atomicInc( volatile long &data )
{
   return ++data;
}

static volatile long s_atomicDec( volatile long &data )
{
   return --data;
}

static void s_lockEngine() {}
static void s_unlockEngine() {}

RWData *rwdata;

}


void EngineData::set() const
{
   memAlloc = m_memAlloc;
   memFree = m_memFree;
   memRealloc = m_memRealloc;
   engineStrings = m_engineStrings;
   Engine::rwdata = m_rwData;
}

void EngineData::initRWData()
{
   Engine::rwdata = new Engine::RWData;
   Engine::rwdata->atomicInc = Engine::s_atomicInc;
   Engine::rwdata->atomicDec = Engine::s_atomicDec;
   Engine::rwdata->lockEngine = Engine::s_lockEngine;
   Engine::rwdata->unlockEngine = Engine::s_unlockEngine;
}

extern "C" void Init( const EngineData &data )
{
   // create the rw data
   data.set();
}


   


}


/* end of enginedata.cpp */
