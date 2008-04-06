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

void EngineData::set() const
{
   memAlloc = m_memAlloc;
   memFree = m_memFree;
   memRealloc = m_memRealloc;
   engineStrings = m_engineStrings;
}

extern "C" void Init(  const EngineData &data )
{
   data.set();
}


}


/* end of enginedata.cpp */
