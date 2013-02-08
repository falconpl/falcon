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

class FalconApp;
class String;
class Log;

class TestMode
{
public:
   TestMode( FalconApp* app );
   ~TestMode();

   class ScriptData {
   public:
      ScriptData( const String& id );
      ~ScriptData();

      String m_id;
      String m_path;

      String m_name;
      String m_category;

      String m_exp_output;
      String m_exp_result;

      bool m_bSuccess;
      String m_reason;
   };

   void setup();
   void perform();
   ScriptData* parse(const String& scriptName );
   void testAll();
   void listAll();
   void test( ScriptData* sd );
   void reportTest( ScriptData* sd );

   void report();

   Log* log;

private:
   FalconApp* m_app;

   typedef std::map<String, ScriptData*> ScriptMap;
   typedef std::map<String, ScriptMap> CategoryMap;

   ScriptMap m_scripts;
   CategoryMap m_categories;
   uint32 m_passed;
};

}

#endif

/* end of testmode.h */
