/*
   FALCON - The Falcon Programming Language.
   FILE: vmtimer.cpp

   Heart beat timer for processors in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 20 Nov 2012 11:41:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vmtimer.h>
#include <map>

namespace Falcon {

class VMTimer::Private {
public:
   typedef std::multimap<int64, Token*> TimeMap;

   TimeMap m_timings;

   Private() {}
   ~Private() {}
};



}
