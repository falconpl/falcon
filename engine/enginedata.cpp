/*
   FALCON - The Falcon Programming Language
   FILE: enginedata.cpp
   $Id: enginedata.cpp,v 1.4 2007/03/05 08:45:28 jonnymind Exp $

   Engine static data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun dic 4 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
