/*
   FALCON - The Falcon Programming Language.
   FILE: testmode.h

   Falcon command line -- Test mode support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 29 Jan 2013 19:12:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_APP_TESTMODE_H
#define FALCON_APP_TESTMODE_H

#include <map>

namespace Falcon {

class TestMode
{
public:
   TestMode( FalconApp* app );
   ~TestMode();

   void setup();
   void perform();
   void test( const String& name );
   void report();

private:
   FalconApp* m_app;

   typedef std::map<String, String> ScriptMap;
   typedef std::map<String, ScriptMap> CategoryMap;

   ScriptMap m_scripts;
   uint32 m_passed;
};


}

#endif

/* end of testmode.h */
