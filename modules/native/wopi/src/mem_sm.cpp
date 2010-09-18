/*
   FALCON - The Falcon Programming Language.
   FILE: mem_sm.cpp

   Falcon Web Oriented Programming Interface

   Memory based session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Mar 2010 10:25:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/mem_sm.h>

namespace Falcon {
namespace WOPI {

//============================================================
// Memory based session data
//

MemSessionData::MemSessionData( const String& SID ):
   SessionData( SID )
{
}

MemSessionData::~MemSessionData()
{
}

bool MemSessionData::resume()
{
   return true;
}

bool MemSessionData::store()
{
   // nothing to do
   return true;
}

bool MemSessionData::dispose()
{
   // nothing to do
   return true;
}

//============================================================
// Memory based session manager
//

MemSessionManager::MemSessionManager()
{
}


MemSessionManager::~MemSessionManager()
{
}

void MemSessionManager::startup()
{
   // nothing to do
}

SessionData* MemSessionManager::createSession( const Falcon::String& sSID )
{
   return new MemSessionData( sSID );
}

}
}

/* end of mem_sm.cpp */
