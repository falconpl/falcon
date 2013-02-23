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
#include <falcon/mt.h>

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

      // long text extension
      int32 m_length;
      int32 m_interval;
      String m_checkpoint;
   };

   void setup();
   void perform();
   ScriptData* parse(const String& scriptName );
   void testAll();
   void listAll();
   void test( ScriptData* sd );
   // returns the full output of the script.
   String* longTest( ScriptData* sd, Process* loadProc );
   void progress( TextWriter& out, ScriptData* sd, int count );

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

   class Reader: public Runnable {
   public:
      Reader();
      virtual ~Reader() {}
      virtual void* run();
      int checkpointCount();
      void setStream( Stream* s ) { m_readStream = s; }
      void setCheckpoint( const String& cp ) { m_checkPoint = cp; }

   private:
      int m_cks;
      mutable Mutex m_mtx;
      Stream* m_readStream;
      String m_checkPoint;
   }
   m_reader;
};

}

#endif

/* end of testmode.h */
