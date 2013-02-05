/*
   FALCON - The Falcon Programming Language.
   FILE: testmode.cpp

   Falcon command line -- Test mode support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 29 Jan 2013 19:12:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/falcon.h>
#include <falcon/modloader.h>
#include <falcon/trace.h>
#include <falcon/log.h>
#include <falcon/stringstream.h>

#include "app.h"
#include "testmode.h"

using namespace Falcon;

TestMode::ScriptData::ScriptData( const String& id ):
         m_id(id)
{
}

TestMode::ScriptData::~ScriptData()
{

}


TestMode::TestMode( FalconApp* app ):
    m_app(app)
{
   log = Engine::instance()->log();
}

TestMode::~TestMode()
{
   ScriptMap::iterator iter = m_scripts.begin();
   ScriptMap::iterator end = m_scripts.end();

   while( iter != end )
   {
      delete iter->second;
      ++iter;
   }

   m_scripts.clear();
}


void TestMode::setup()
{
   // TODO: Fill the categories.
}

void TestMode::perform()
{
   setup();

   try
   {
      Directory* dir = Engine::instance()->vfs().openDir( m_app->m_options.test_dir );

      String name;

      // first, we build a script map, so we can run tests in order.
      while(dir->read(name))
      {
         // Check format "NNN-NNN.fal"
         if( name.size() == 11 && name.subString(7,11) == ".fal")
         {
            for( int i = 0; i < 7; ++i )
            {
               char n = name.getCharAt(i);

               if( (i == 3 && n != '-') ||( i != 3 && ( n < '0' || n > '9' )))
               {
                  continue;
               }
            }
         }
         else  {
            continue;
         }

         if( m_app->m_options.test_category != "" )
         {
            if( ! name.startsWith(m_app->m_options.test_category) )
            {
               log->log( Log::fac_app, Log::lvl_debug, String( "Skipping because not in selected category " ) + name );
               continue;
            }
         }

         // ok, file in correct format.
         try {
            ScriptData* sd = parse( name );
            m_scripts[name] = sd;
            log->log( Log::fac_app, Log::lvl_detail, String( "Adding script " ) + name );
         }
         catch (Error* error)
         {
            log->log( Log::fac_app, Log::lvl_warn, String( "Can't read script " ) + name + ":" + error->describe() );
            error->decref();
         }
      }

      if( m_scripts.size() != 0 )
      {
         log->log( Log::fac_app, Log::lvl_info, String( "Starting to test " ).N(m_scripts.size()).A( " scripts") );
      }
      else {
         log->log( Log::fac_app, Log::lvl_warn, String( "No script in format NNN-NNN.fal found in target directory.") );
      }

      testAll();
      report();
   }
   catch(Error* e)
   {
      // let the main catcher to do the logging.
      throw e;
   }
}


void TestMode::report()
{
   TextWriter ts( new StdOutStream );
   ts.writeLine( String("Complete. Passed ").N(m_passed).A("/").N( m_scripts.size() ) );
}


void TestMode::testAll()
{
   ScriptMap::iterator iter = m_scripts.begin();
   ScriptMap::iterator end = m_scripts.end();

   m_passed = 0;
   while( iter != end )
   {
      ScriptData* sd = iter->second;
      test( sd );

      // perform a full GC
      log->log( Log::fac_app, Log::lvl_detail, String( "Starting full GC." ) );
      Engine::instance()->collector()->performGC(true);
      log->log( Log::fac_app, Log::lvl_info, String( "Full GC complete." ) );

      reportTest(sd);
      ++iter;
   }
}


TestMode::ScriptData* TestMode::parse(const String& scriptName )
{
   ScriptData* sd = new ScriptData(scriptName.subString(0,7));
   String fname = m_app->m_options.test_dir + "/" + scriptName;
   sd->m_path = fname;

   bool gettingOut = false;
   try
   {

      Stream* input = Engine::instance()->vfs().openRO( fname );
      input->shouldThrow(true);
      TextReader tr(input, true);
      String line;

      while( ! tr.eof() )
      {
         tr.readLine(line,256);
         String tline = line;

         if( gettingOut )
         {
            if( tline == "@endoutput" )
            {
               gettingOut = false;
            }
            else
            {
               if( sd->m_exp_output.size() != 0 )
               {
                  sd->m_exp_output +="\n";
               }

               sd->m_exp_output += tline;
            }
         }
         else
         {
            tline.trim();

            if( tline.startsWith("@name ") )
            {
               sd->m_name = tline.subString(5);
               sd->m_name.trim();
            }
            else if( tline == ("@output") )
            {
               sd->m_exp_output = "";
               gettingOut = true;
            }
            else if( tline.startsWith("@result ") )
            {
               sd->m_exp_result = tline.subString(8);
               sd->m_exp_result.trim();
            }
         }
      }
   }
   catch(...)
   {
      delete sd;
      throw;
   }

   return sd;
}


void TestMode::test( ScriptData* sd )
{
   log->log( Log::fac_app, Log::lvl_info, String( "Now testing " ) + sd->m_id );

   VMachine vm;
   // capture VM output
   StringStream* ss = new StringStream;
   vm.stdOut( ss );

   m_app->configureVM(vm);
   Process* loadProc = vm.modSpace()->loadModule( sd->m_path, true, true, false );

   try {
      loadProc->start();
      loadProc->wait();

      sd->m_bSuccess = true;

      if( sd->m_exp_output.size() != 0 )
      {
         String* result = ss->closeToString();
         if ( sd->m_exp_output != *result && (sd->m_exp_output+"\n") != *result )
         {
            sd->m_bSuccess = false;
            sd->m_reason = "Expected output not matching";
            log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                                    +" failing because expected output is not matching" );
         }

         delete result;
      }

      if( sd->m_bSuccess == true )
      {
         if( sd->m_exp_result.size() != 0 )
         {
            const Item& result = loadProc->result();
            if( result.isString() )
            {
               if( sd->m_exp_result != *result.asString() )
               {
                  sd->m_bSuccess = false;
                  sd->m_reason = "Expected final result not matching";
                  log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                           +" failing because " + sd->m_exp_result+ "!=" + *result.asString() );
               }
            }
            else {
               sd->m_bSuccess = false;
               sd->m_reason = "Returned final result is not a string";
               log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                        +" failing because final result is not a string" );
            }
         }
      }

      if( sd->m_bSuccess )
      {
         log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                                 +" success ");
      }
   }
   catch( Error* err )
   {
      // explicit termination reason?
      sd->m_bSuccess = false;
      if( err->raised().isString() )
      {
         sd->m_reason = *err->raised().asString();
      }
      else {
         sd->m_reason = "Terminated with error: ";
         sd->m_reason += err->describe();
         log->log( Log::fac_app, Log::lvl_warn, String("Test ") +sd->m_id
                                 +" terminated with error: "+ sd->m_reason );
      }
      err->decref();
   }

   loadProc->decref();
}

void TestMode::reportTest( ScriptData* sd )
{
   TextWriter output(new StdOutStream(true), true);

   output.write( sd->m_id + ": ");
   if( sd->m_bSuccess )
   {
      output.writeLine( "Success" );
      m_passed ++;
   }
   else {
      output.writeLine( "Fail (" + sd->m_reason + ")" );
   }
   output.flush();
}

/* end of testmode.cpp */

