/*
   FALCON - The Falcon Programming Language.
   FILE: mem_sm.h

   Falcon Web Oriented Programming Interface

   Memory based session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Mar 2010 10:25:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _MEM_SESSION_MANAGER_H
#define _MEM_SESSION_MANAGER_H

#include <falcon/wopi/session_manager.h>

namespace Falcon {
namespace WOPI {


/** Memory based session data.

    This session data doesn't provide any form of persistency.
*/
class MemSessionData: public SessionData
{
public:
   MemSessionData( const String& SID );
   virtual ~MemSessionData();

   virtual bool resume();
   virtual bool store();
   virtual bool dispose();

};


/** Memory based session manager.
    This session store all the data about session just in memory.

    Client code restart will clear all the sessions (without any information
    about the fact that existing sessions are invalidated).
*/

class MemSessionManager: public SessionManager
{
public:
   MemSessionManager();
   virtual ~MemSessionManager();

   virtual void startup();

protected:
   virtual SessionData* createSession( const String& sSID );

};

}
}

#endif

/* end of mem_sm.h */
