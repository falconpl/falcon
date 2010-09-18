/*
   FALCON - The Falcon Programming Language.
   FILE: file_sm.h

   Falcon Web Oriented Programming Interface

   File based session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 27 Mar 2010 15:00:17 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FILE_SESSION_MANAGER_H
#define _FILE_SESSION_MANAGER_H

#include <falcon/wopi/session_manager.h>

namespace Falcon {
namespace WOPI {


/** File based session data.
*/
class FileSessionData: public SessionData
{
public:
   FileSessionData( const String& SID, const String& tmpDir );
   virtual ~FileSessionData();

   virtual bool resume();
   virtual bool store();
   virtual bool dispose();

private:
   String m_tmpDir;
};


/** File based session manager.
*/

class FileSessionManager: public SessionManager
{
public:
   FileSessionManager( const String& tmpDir );
   virtual ~FileSessionManager();

   virtual void startup();
   virtual void configFromModule( const Module* mod );
protected:
   virtual SessionData* createSession( const String& sSID );

private:
   String m_tmpDir;

};

}
}

#endif

/* end of file_sm.h */
