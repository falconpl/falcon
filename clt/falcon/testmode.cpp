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
#undef SRC
#define SRC "testmode.cpp"

#include <falcon/falcon.h>
#include <falcon/modloader.h>
#include <falcon/trace.h>
#include <falcon/log.h>
#include <falcon/stringstream.h>

#include <falcon/pipe.h>
#include <falcon/textwriter.h>
#include <falcon/textreader.h>
#include <falcon/mt.h>
#include <falcon/errors/genericerror.h>

#include <falcon/sys.h>

#include "app.h"
#include "testmode.h"

using namespace Falcon;

static void stripReturns( String* result )
{
   uint32 pos = result->find('\r');
   while( pos != String::npos )
   {
      result->remove(pos,1);
      pos = result->find('\r', pos);
   }
}

TestMode::ScriptData::ScriptData( const String& id ):
         m_id(id),
         m_length(0),
         m_interval(0)
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
         if( name.length() == 11 && name.subString(7,11) == ".fal")
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

         if( m_app->m_options.test_prefix != "" )
         {
            if( ! name.startsWith(m_app->m_options.test_prefix) )
            {
               log->log( Log::fac_app, Log::lvl_debug, String( "Skipping because not in selected category " ) + name );
               continue;
            }
         }

         // ok, file in correct format.
         try {
            ScriptData* sd = parse( name );
            if( m_app->m_options.test_category == ""
                ||  m_app->m_options.test_category == sd->m_category )
            {
               m_scripts[name] = sd;
               ScriptMap& map = m_categories[sd->m_category];
               map[name] = sd;
            }

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

      if( m_app->m_options.list_tests )
      {
         listAll();
      }
      else {
         testAll();
         report();
      }
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



void TestMode::listAll()
{
   ScriptMap::iterator iter = m_scripts.begin();
   ScriptMap::iterator end = m_scripts.end();

   TextWriter output(new StdOutStream(true));
   output.setSysCRLF();
   if( iter == end )
   {
      output.writeLine( "The filter has determined an empty list" );
      return;
   }

   while( iter != end )
   {
      ScriptData* sd = iter->second;
      String category = sd->m_category == "" ? "uncategorized" : sd->m_category;

      output.writeLine( sd->m_id + " (" + category+ "): " + sd->m_name );

      ++iter;
   }

   output.flush();
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
      TextReader tr(input);
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
            else if( tline.startsWith("@category ") )
            {
               sd->m_category = tline.subString(10);
               sd->m_category.trim();
            }
            else if( tline.startsWith("@long ") )
            {
               sd->m_length = atoi( tline.subString(6).c_ize() );
            }
            else if( tline.startsWith("@interval ") )
            {
               sd->m_interval = atoi( tline.subString(10).c_ize() );
            }
            else if( tline.startsWith("@checkpoint ") )
            {
               sd->m_checkpoint = tline.subString(12);
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

   Process* loadProc = vm.createProcess();
   m_app->configureVM( vm, loadProc );

   ModSpace* ms = loadProc->modSpace();
   ms->loadModuleInProcess( sd->m_path,  true, true, false );


   try
   {
      String* result = 0;
      // suppose true
      sd->m_bSuccess = true;

      if( sd->m_length != 0 )
      {
         result = longTest( sd, loadProc );
      }
      else
      {
         // capture VM output
         StringStream* ss = new StringStream;
         vm.stdOut( ss );

         loadProc->start();
         loadProc->wait();
         if( ! sd->m_exp_output.empty() )
         {
            result = ss->closeToString();
         }
      }

      if( sd->m_bSuccess && sd->m_exp_output.size() != 0 )
      {
         // neutralize xplatform \r\n things.
         stripReturns( result );

         if ( sd->m_exp_output != *result && (sd->m_exp_output+"\n") != *result )
         {
            sd->m_bSuccess = false;
            sd->m_reason = "Expected output not matching";
            log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                                    +" failing because expected output is not matching" );
         }

         delete result;
      }

      if( sd->m_bSuccess )
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



String* TestMode::longTest( ScriptData* sd, Process* loadProc )
{
#define TestMode_READER_THREAD_STACK_SIZE 20384

   Stream::L stream(new StdOutStream(true));
   TextWriter output( stream );
   log->log( Log::fac_app, Log::lvl_info, String( "Entering long test for " ) + sd->m_id );

   // redirect the VM output to a pipe we can control.
   Sys::Pipe controlPipe;
   Stream* writeStream = controlPipe.getWriteStream();
   loadProc->vm()->stdOut( writeStream );

   SysThread* thread = new SysThread( &m_reader );
   m_reader.setStream( controlPipe.getReadStream() );
   m_reader.setCheckpoint( sd->m_checkpoint );
   if ( ! thread->start(ThreadParams().stackSize(TestMode_READER_THREAD_STACK_SIZE)) )
   {
      throw new GenericError(ErrorParam( e_io_error,  __LINE__, SRC )
               .extra( "Starting thread"));
   }

   loadProc->start();

   int oldCpcount = 0;
   int64 timeout = sd->m_interval > 0 ? (Sys::_milliseconds() + sd->m_interval ) : -1;

   while( true )
   {
      int cpcount = m_reader.checkpointCount();

      progress( output, sd, cpcount );
      // was there real progress?
      if( cpcount != oldCpcount)
      {
         oldCpcount = cpcount;
         if( timeout > 0 )
         {
            // reset the timeout
            timeout = Sys::_milliseconds() + sd->m_interval;
         }
      }

      try
      {
         // too slow?
         if( timeout > 0 && Sys::_milliseconds() > timeout )
         {
            loadProc->terminate();
            loadProc->wait();
            sd->m_bSuccess = false;
            sd->m_reason = String("Timeout of ").N(sd->m_interval).A("ms elapsed at step ")
                     .N( cpcount ).A(" - ").N((cpcount*100.0)/sd->m_length, "%02.2f").A("%");

            log->log( Log::fac_app, Log::lvl_info, String("Test ") +sd->m_id
                                             +" failure: " + sd->m_reason );
            break;
         }

         // normal wait loop
         if( loadProc->wait(50) )
         {
            break;
         }
      }
      catch( ... )
      {
         writeStream->close();
         void* dummy = 0;
         thread->join( dummy );
         delete static_cast<String*>(dummy);
         output.write(String(" ").replicate(78).A("\r"));
         output.flush();
         throw;
      }
   }

   // a last progress before flashing the close
   progress( output, sd, m_reader.checkpointCount() );
   // get the output from the reader.
   writeStream->close();
   void* scriptOutput = 0;
   thread->join( scriptOutput );

   // clear the line
   output.write(String(" ").replicate(78).A("\r"));
   output.flush();

   return static_cast<String*>(scriptOutput);
}


void TestMode::progress( TextWriter& output, ScriptData* sd, int count )
{
   #define TEST_PROGRESS_BAR_LENGTH 50

   static int flapPos = 0;
   const char* flaps ="\\|/-";

   output.write(sd->m_id);
   output.write(": [");
   output.write(String().A(flaps[flapPos++]));
   if( flapPos == 4 ) flapPos = 0;
   output.write("] ");

   double pct = (count * 100.0) / sd->m_length;
   int filled = (int) (TEST_PROGRESS_BAR_LENGTH * (pct/100.0));
   if( filled > 0  )
   {
      if( filled > TEST_PROGRESS_BAR_LENGTH ) filled = TEST_PROGRESS_BAR_LENGTH;
      output.write(String("=").replicate(filled));
   }

   output.write(String(" ").replicate(TEST_PROGRESS_BAR_LENGTH - filled));
   output.write(" ");
   output.write(String().N(pct, "%02.2f").A("%   \r"));
   output.flush();
}


void TestMode::reportTest( ScriptData* sd )
{
   Stream::L stream(new StdOutStream(true));
   TextWriter output(stream);
   output.setSysCRLF();

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


TestMode::Reader::Reader():
      m_cks(0),
      m_readStream(0)
{}

void* TestMode::Reader::run()
{
   fassert( m_readStream !=0 );
   TextReader reader( m_readStream );
   m_cks = 0;

   String* output = new String;

   String line;
   while(reader.readLine(line, 4096) )
   {
      if( m_checkPoint.empty() || line == m_checkPoint )
      {
         m_mtx.lock();
         m_cks++;
         m_mtx.unlock();
      }

      output->append(line);
      output->append('\n');
   }

   return output;
}


int TestMode::Reader::checkpointCount()
{
   m_mtx.lock();
   int ret = m_cks;
   m_mtx.unlock();

   return ret;
}

/* end of testmode.cpp */

