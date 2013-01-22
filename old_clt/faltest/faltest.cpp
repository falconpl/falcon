/*
   FALCON - The Falcon Programming Language.
   FILE: faltest.cpp

   Testsuite interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This program embeds the Falcon compiler and perform tests on all the
   files contained in a certain directory.

   It is meant to test Falcon features in case of changes and similar.
*/

#include <falcon/setup.h>
#include <stdlib.h>

#include <falcon/sys.h>
#include <falcon/setup.h>
#include <falcon/common.h>
#include <falcon/compiler.h>
#include <falcon/genhasm.h>
#include <falcon/gencode.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/core_ext.h>
#include <falcon/modloader.h>
#include <falcon/string.h>
#include <falcon/testsuite.h>
#include <falcon/memory.h>
#include <falcon/fstream.h>
#include <falcon/stringstream.h>
#include <falcon/stdstreams.h>
#include <falcon/transcoding.h>
#include <falcon/path.h>
#include <falcon/signals.h>

#if FALCON_LITTLE_ENDIAN != 1
#include <falcon/pcode.h>
#endif

#include "scriptdata.h"

#define DEF_PREC  5
#define TIME_PRINT_FMT "%.3f"

using namespace Falcon;

/************************************************
   Global data and options
*************************************************/

bool opt_compmem;
bool opt_compasm;
bool opt_justlist;
bool opt_verbose;
bool opt_serialize;
bool opt_timings;
bool opt_inTimings;
bool opt_checkmem;
String opt_output;
String opt_category;
String opt_subcat;
String opt_path;
String opt_libpath;
int opt_tf;
Stream *output;

int passedCount;
int failedCount;
int totalCount;

bool testStatus;
String failureReason;

double total_time_compile;
double total_time_generate;
double total_time_link;
double total_time_execute;

double compTime;
double genTime;
double linkTime;
double execTime;

Falcon::List opt_testList;

/** Main script dictionary */
t_idScriptMap scriptMap;

/** Helper dictionary to help categorization */
t_categoryScriptMap categoryMap;

/** Standard output stream */
Stream *stdOut;
Stream *stdErr;


/************************************************
   Functions used to account memory management
*************************************************/

static long s_allocatedMem = 0;
static long s_totalMem = 0;
static long s_outBlocks = 0;
static long s_totalOutBlocks = 0;


/************************************************
   Typical utility functions for command lines
*************************************************/

static void version()
{
   stdOut->writeString( "Falcon unit test package.\n" );
   stdOut->writeString( "Version " );
   stdOut->writeString( FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   stdOut->writeString( "\n" );
   stdOut->flush();
}

static void usage()
{
   stdOut->writeString( "Usage: faltest [options] -d testsuite_directory [tests ids]\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Options:\n" );
   stdOut->writeString( "   -c <cat>    only perform tests in this category\n" );
   stdOut->writeString( "   -C <subcat> only perform test in this subcategory\n" );
   stdOut->writeString( "   -d <path>   tests are in specified directory\n" );
   stdOut->writeString( "   -l          Just list available tests and exit\n" );
   stdOut->writeString( "   -L          Changes Falcon load path.\n" );
   stdOut->writeString( "   -h/-?       Show this help\n" );
   stdOut->writeString( "   -m          do NOT compile in memory\n" );
   stdOut->writeString( "   -M          Check for memory allocation correctness.\n" );
   stdOut->writeString( "   -f <n>      set time factor to N for benchmarks\n" );
   stdOut->writeString( "   -o <file>   Output report here (defaults stdout)\n" );
   stdOut->writeString( "   -s          perform module serialization test\n" );
   stdOut->writeString( "   -S          compile via assembly\n" );
   stdOut->writeString( "   -t          record and display timings\n" );
   stdOut->writeString( "   -T          read internal script timing() request\n" );
   stdOut->writeString( "   -v          be verbose\n" );
   stdOut->writeString( "   -V          print copyright notice and version and exit\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Path must be in falcon file name format: directory separatros must be slashes.\n" );
   stdOut->writeString( "A list of space separated tests IDS to be performed may optionally be given.\n" );
   stdOut->writeString( "\n" );
   stdOut->flush();
}


void parse_options( int argc, char *argv[] )
{
   opt_compmem = true;
   opt_checkmem = false;
   opt_compasm = false;
   opt_justlist = false;
   opt_verbose = false;
   opt_serialize = false;
   opt_timings = false;
   opt_inTimings = false;
   opt_tf = 1;

   // option decoding
   for ( int i = 1; i < argc; i++ )
   {
      char *op = argv[i];

      if (op[0] == '-' )
      {
         switch ( op[1] )
         {
            case 'c':
               if( op[2] != 0 )
                  opt_category = op + 2;
               else if ( i < argc - 1 ) {
                  i++;
                  opt_category = argv[i];
               }
               else {
                  stdOut->writeString( "Must specify an argument for -c\n" );
                  usage();
                  exit(1);
               }
            break;

            case 'C':
               if( op[2] != 0 )
                  opt_subcat = op + 2;
               else if ( i < argc - 1 ) {
                  i++;
                  opt_subcat = argv[i];
               }
               else {
                  stdOut->writeString( "Must specify an argument for -C\n" );
                  usage();
                  exit(1);
               }
            break;

            case 'd':
               if( op[2] != 0 )
                  opt_path = op + 2;
               else if ( i < argc - 1 ) {
                  i++;
                  opt_path = argv[i];
               }
               else {
                  stdOut->writeString( "Must specify an argument for -d\n" );
                  usage();
                  exit(1);
               }
            break;

            case 'L':
               if( op[2] != 0 )
                  opt_libpath = op + 2;
               else if ( i < argc - 1 ) {
                  i++;
                  opt_libpath = argv[i];
               }
               else {
                  stdOut->writeString( "Must specify an argument for -L\n" );
                  usage();
                  exit(1);
               }
            break;

            case '?': case 'h': usage(); exit(0);
            case 'l': opt_justlist = true; break;
            case 'm': opt_compmem = false; break;
            case 'M': opt_checkmem = true; break;

            case 'o':
               if( op[2] != 0 )
                  opt_output = op + 2;
               else if ( i < argc - 1 ) {
                  i++;
                  opt_output = argv[i];
               }
               else {
                  stdOut->writeString( "Must specify an argument for -o\n" );
                  usage();
                  exit(1);
               }
            break;

            case 's': opt_serialize = true; break;
            case 'S': opt_compasm = true; break;
            case 't': opt_timings = true; break;
            case 'f':
              if( op[2] != 0 )
                  opt_tf = atoi(op + 2);
               else if ( i < argc - 1 ) {
                  i++;
                  opt_tf = atoi(argv[i]);
               }
               else {
                  stdOut->writeString( "Must specify an argument for -f\n" );
                  usage();
                  exit(1);
               }
               if (opt_tf <= 0)
                  opt_tf = 1;
            break;

            case 'T': opt_inTimings = true; break;
            case 'v': opt_verbose = true; break;

            case 'V': exit(0);
            default:
               stdErr->writeString( "falcon: unrecognized option '" );
               stdErr->writeString( op );
               stdErr->writeString( "'.\n\n" );
               usage();
               exit(0);
         }
      }
      else {
         opt_testList.pushBack( new String(argv[i]) );
      }
   }
}

bool readline( Stream &script, String &line )
{
   line = "";
   char c;
   while( script.read( &c, 1 ) == 1 )
   {
      if ( c == '\n' )
         return true;
      if ( c != '\r' )
         line += c;
   }
   return false;
}

/************************************************
   Read script definitions
*************************************************/
ScriptData *readProperties( const String &name, Stream &script )
{
   // let's get the script props.
   ScriptData *data = new ScriptData( name );
   String line;
   int mode = 0;
   int descrMode = 0;
   String description;

   while ( readline( script, line ) )
   {
      if ( mode == 0 )
      {
         if ( line.find( "/*" ) == 0 )
            mode = 1;
      }
      else {
         if ( line.find( "*/" ) == 0 )
         {
            mode = 0;
            if ( descrMode != 0 ) {
               data->setProperty( "Description", description );
               description = "";
               descrMode = 0;
            }
         }
         else
         {
            if (line.find( "*" ) == 0 )
            {
               if ( descrMode == 1 )
               {
                  int beg = 1;
                  while ( line.getCharAt( beg ) == ' ' ) beg++;
                  line = line.subString( beg );
                  if ( line == "[/Description]" )
                  {
                     data->setProperty( "Description", description );
                     descrMode = 0;
                     description = "";
                  }
                  else {
                     description += line;
                     description += '\n';
                  }
               }
               else {
                  uint32 limit = line.find( ":" );
                  if ( limit != String::npos )
                  {
                     // ok, we have a property
                     int beg = 1;
                     while ( line.getCharAt( beg ) == ' ' ) beg++;
                     String prop = line.subString( beg, limit );
                     limit++;
                     while ( line.getCharAt( limit ) == ' ' ) limit++;
                     String val = line.subString(limit);
                     if ( prop == "ID" ) {
                        if ( val.length() && val.getCharAt( 0 ) != '-' )
                        {
                           data->id( ScriptData::IdCodeToId( val ) );
                        }
                        else {
                           data->id( -1 );
                           break;
                        }
                     }

                     if ( prop == "Description" )
                     {
                        descrMode = 1;
                     }
                     else
                        data->setProperty( prop, val );
                  }
               }
            }
         }
      }
   }

   // invalid data?
   if ( data->id() == -1 )
   {
      delete data;
      data = 0;
   }

   return data;
}


/************************************************
   Script descriptions
*************************************************/
void describeScript( ScriptData *data )
{

   String idCode;
   ScriptData::IdToIdCode( data->id(), idCode );
   String val;
   output->writeString( idCode + " - " );
   data->getProperty( "Short", val );
   output->writeString( val );

   if ( ! opt_verbose )
   {
      output->writeString( " - " + data->filename() + " - " );
      if ( data->getProperty( "Category", val ) )
      {
         output->writeString( " (" + val );
         if( data->getProperty( "Subcategory", val ) )
            output->writeString( "/" + val );
         output->writeString( ")" );
      }
      output->writeString( "\n" );
   }
   else
   {
      output->writeString( "\n" );
      output->writeString( "Script file: " + opt_path + "/" );
      output->writeString( data->filename() + "\n" );

      String cat;

      if ( data->getProperty( "Category", cat ) )
         output->writeString( "Category: " + cat + "\n" );

      if ( data->getProperty( "Subcategory", cat ) )
         output->writeString( "Subcategory: " + cat + "\n" );

      data->getProperty( "Description", val );
      output->writeString( val + "\n" );
   }
}


void listScripts()
{
   if ( opt_verbose )
   {
      output->writeString( "-----------------------------------------------------------\n" );
   }
   if ( opt_testList.empty() )
   {
      t_idScriptMap::const_iterator iter = scriptMap.begin();
      while( iter !=  scriptMap.end() )
      {
         describeScript( iter->second );
         if ( opt_verbose )
         {
            output->writeString( "-----------------------------------------------------------\n" );
         }
         ++iter;
      }
   }
   else {
      ListElement *iter = opt_testList.begin();
      while ( iter != 0 )
      {
         const String &name = *(String *) iter->data();
         t_idScriptMap::const_iterator elem =
            scriptMap.find( ScriptData::IdCodeToId( name ) );
         if ( elem == scriptMap.end() )
            output->writeString( name + " - NOT FOUND\n" );
         else {
            describeScript( elem->second );
            if ( opt_verbose )
            {
               output->writeString( "-----------------------------------------------------------\n" );
            }
         }
         iter = iter->next();
      }
   }
}

void filterScripts()
{
   t_idScriptMap filterMap;

   if ( opt_testList.empty() )
   {
      t_idScriptMap::iterator iter = scriptMap.begin();
      while( iter !=  scriptMap.end() )
      {
         String cat, subcat;
         ScriptData *script = iter->second;
         script->getProperty( "Category", cat );
         script->getProperty( "Subcategory", subcat );

         if ( ( opt_category == "" || cat == opt_category )&&
               ( opt_subcat == "" || subcat == opt_subcat ) )
         {
            filterMap[iter->first] = iter->second;
            iter->second = 0;
         }
         ++iter;
      }
   }
   else {
      ListElement *iter = opt_testList.begin();
      while ( iter != 0 )
      {
			const String &name = *(String *) iter->data();
         t_idScriptMap::iterator elem =
            scriptMap.find( ScriptData::IdCodeToId( name ) );
         if ( elem != scriptMap.end() )
         {
            filterMap[ elem->first ] = elem->second;
            elem->second = 0;
         }
         iter = iter->next();
      }
   }

   t_idScriptMap::iterator liter = scriptMap.begin();
   while( liter != scriptMap.end() )
   {
      delete liter->second;
      ++liter;
   }
   scriptMap = filterMap;
}

/************************************************
   Script testing
*************************************************/
bool testScript( ScriptData *script,
         ModuleLoader *modloader, Module *core, Module *testSuite,
         String &reason, String &trace )
{
   Module *scriptModule;

   //---------------------------------
   // 1. compile
   String path = opt_path + "/" + script->filename();

   FileStream *source_f = new FileStream;

   source_f->open( path );
   if( ! source_f->good() )
   {
      reason = "Can't open source " + path;
      delete source_f;
      return false;
   }

   Stream *source = TranscoderFactory( "utf-8", source_f, true );

   scriptModule = new Module();
   Path scriptPath( script->filename() );
   scriptModule->name( scriptPath.getFile() );
   scriptModule->path( path );

   Compiler compiler( scriptModule, source );
   compiler.searchPath( Engine::getSearchPath() );

   if ( opt_timings )
      compTime = Sys::_seconds();

   if ( ! compiler.compile() )
   {
      Error* err = compiler.detachErrors();
      trace = err->toString();
      err->decref();
      reason = "Compile step failed.";
      delete source;
      scriptModule->decref();
      return false;
   }

   if ( opt_timings )
      compTime = Sys::_seconds() - compTime;

   // we can get rid of the source here.
   delete source;

   // now compile the code.
   GenCode gc( compiler.module() );
   if ( opt_timings )
      genTime = Sys::_seconds();
   gc.generate( compiler.sourceTree() );

   if ( opt_timings )
      genTime = Sys::_seconds() - genTime;

   // serialization/deserialization test
   if( opt_serialize )
   {
      // create the module stream
      FileStream *module_stream = new FileStream;
      // create a temporary file
      module_stream->create( "temp_module.fam",  FileStream::e_aUserWrite | FileStream::e_aReadOnly );

      scriptModule->save( module_stream, false );
      module_stream->seekBegin( 0 );
      String scriptName = scriptModule->name();
      scriptModule->decref();
      scriptModule = new Module();
      scriptModule->name( scriptName );
      scriptModule->path( path );

      if ( ! scriptModule->load( module_stream, false ) )
      {
         reason = "Deserialization step failed.";
         delete module_stream;
         scriptModule->decref();
         return false;
      }

      delete module_stream;
   }

   //---------------------------------
   // 2. link
   VMachineWrapper vmachine;

   // so we can link them
   vmachine->link( core );
   vmachine->link( testSuite );
   Runtime runtime( modloader );

   try
   {
      runtime.addModule( scriptModule );
   }
   catch (Error *err)
   {
      trace = err->toString();
      err->decref();
      reason = "Module loading failed.";
      scriptModule->decref();
      return false;

   }

   // we can abandon our reference to the script module
   scriptModule->decref();

   //---------------------------------
   // 3. execute
   TestSuite::setSuccess( true );
   TestSuite::setTimeFactor( opt_tf );
   if ( opt_timings )
         execTime = Sys::_seconds();

   // inject args and script name
   Item *sname =vmachine->findGlobalItem( "scriptName" );
   *sname = new CoreString( scriptModule->name() );
   sname =vmachine->findGlobalItem( "scriptPath" );
   *sname = new CoreString( scriptModule->path() );

   try
   {
     vmachine->link( &runtime );
   }
   catch( Error *err )
   {
      trace = err->toString();
      err->decref();
      reason = "VM Link step failed.";
      return false;
   }

   if ( opt_timings )
      linkTime = Sys::_seconds();

   // Become target of the OS signals.
   // vmachine->becomeSignalTarget();

   try
   {
      try
      {
         vmachine->launch();
      }
      catch( CodeError* err )
      {
         err->decref();
         trace = "";
         reason = "Non executable script.";
         return false;
      }

      if ( opt_timings )
         execTime = Sys::_seconds() - execTime;
   }
   catch( Error *err )
   {

      trace = err->toString();
      err->decref();
      reason = "Abnormal script termniation";
      return false;
   }

   // get timings
   if ( opt_timings ) {
      total_time_compile += compTime;
      total_time_execute += execTime;
      total_time_link += linkTime;
      total_time_generate += genTime;
   }

   // reset the success status
   reason = TestSuite::getFailureReason();
   // ensure to clean memory -- do this now to ensure the VM is killed before we kill the module.
   modloader->compiler().reset();
   return TestSuite::getSuccess();
}


void gauge()
{
   if ( opt_output != "" )
   {
      double ratio = (passedCount + failedCount) / double( scriptMap.size() );
      int gaugesize = (int) (50 * ratio);
      stdOut->writeString( "\r[" );
      for ( int i = 0; i < gaugesize; i++ )
         stdOut->writeString( "#" );
      for ( int j = gaugesize; j < 50; j++ )
         stdOut->writeString( " " );

      String temp = "] ";
      temp.writeNumber( (int64)(ratio *100) );
      temp += "% (";
      temp.writeNumber( (int64) passedCount + failedCount );
      temp += "/";
      temp.writeNumber( (int64) scriptMap.size() );
      temp += ")";

      if ( failedCount > 0 )
         stdOut->writeString( String(" fail ").A(failedCount) );
   }
}

void executeTests( ModuleLoader *modloader )
{
   Module *core = Falcon::core_module_init();
   Module *testSuite = init_testsuite_module();

   // now we are all armed up.
   t_idScriptMap::const_iterator iter;
   ListElement *order = 0;

   if( opt_testList.empty() )
      iter = scriptMap.begin();
   else {
      order = opt_testList.begin();
      iter = scriptMap.end();
      while( iter == scriptMap.end() && order != 0 ) {
         const String &name = *(String *) order->data();
         iter = scriptMap.find( ScriptData::IdCodeToId( name ) );
         order = order->next();
      }
   }

   while( iter != scriptMap.end() )
   {
      String cat, subcat;
      ScriptData *script = iter->second;

      String reason, trace;
      // ... we use reason as a temporary storage...
      ScriptData::IdToIdCode( script->id(), reason );
      TestSuite::setTestName( reason );
      output->writeString( reason + ": " );
      output->flush();

      // ... and clear it before getting memory allocation
      reason = "";

      if ( opt_checkmem )
      {
         s_allocatedMem = gcmallocated();
         s_outBlocks = memPool->allocatedItems();
      }

      bool success = testScript( script, modloader, core, testSuite, reason, trace );

      if ( success )
      {
         passedCount++;
         output->writeString( "success." );
         if ( opt_timings ) {
            String temp = "(";
            temp.writeNumber( compTime, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( genTime, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( linkTime, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( execTime, TIME_PRINT_FMT );
            temp += ")";
            output->writeString( temp );
         }
         if ( opt_inTimings )
         {
            numeric tott, opNum;
            TestSuite::getTimings( tott, opNum );
            if (opNum > 0.0 && tott >0.0)
            {
               String temp = " ( ";
               temp.writeNumber( tott, TIME_PRINT_FMT );
               temp += " secs, ";
               temp.writeNumber( opNum / tott, TIME_PRINT_FMT );
               temp += " ops/sec)";
               output->writeString( temp );
            }
         }

         if ( opt_checkmem )
         {
            memPool->performGC();
            long alloc = gcmallocated() - s_allocatedMem;
            long blocks = memPool->allocatedItems() - s_outBlocks;

            String temp = " (leak: ";
            temp.writeNumber( (int64) alloc );
            temp += "/";
            temp.writeNumber( (int64) blocks );
            temp += ")";
            output->writeString( temp );
         }

         output->writeString( "\n" );
      }
      else {
         failedCount++;
         if( opt_verbose ) {
            output->writeString( "fail.\n" );
            describeScript( script );

            output->writeString( "Reason: " + reason + "\n" );
            if ( trace != "" )
            {
               output->writeString( "Error trace: \n------------ \n\n" );
               output->writeString( trace );
               output->writeString( "\n" );
            }
         }
         else
            output->writeString( "fail. (" + reason + ")\n" );
      }

      if ( opt_verbose ) {
         output->writeString( "-------------------------------------------------------------\n\n");
      }

      gauge();

      if( opt_testList.empty() )
         ++iter;
      else {
         iter = scriptMap.end();
         while( iter == scriptMap.end() && order != 0 ) {
				const String &name = *(String *) order->data();
            iter = scriptMap.find(  ScriptData::IdCodeToId( name ) );
            order = order->next();
         }
      }
   }
   // clear the testname (or we'll have 1 extra block in accounts)
   TestSuite::setTestName( "" );

   core->decref();
   testSuite->decref();
}

/************************************************
   Main function
*************************************************/

int main( int argc, char *argv[] )
{
   // Install a void ctrl-c handler (let ctrl-c to kill this app)
   Sys::_dummy_ctrl_c_handler();

   /* Block signals same way falcon binary does. */
   BlockSignals();

   Falcon::Engine::Init();

   stdOut = stdOutputStream();
   stdErr = stdErrorStream();

   // setting an environment var for self-configurating scripts
   Sys::_setEnv( "FALCON_TESTING", "1" );

   version();
   parse_options( argc, argv );

   FileStream *fs_out = new FileStream;
   passedCount = 0;
   failedCount = 0;
   totalCount = 0;
   total_time_compile = 0.0;
   total_time_generate = 0.0;
   total_time_link = 0.0;
   total_time_execute = 0.0;

   double appTime;
   if( opt_timings )
      appTime = Sys::_seconds();
   else
      appTime = 0.0;

   ModuleLoader *modloader = new ModuleLoader( opt_libpath == "" ? "." : opt_libpath );
   modloader->addFalconPath();
   Engine::setSearchPath( modloader->getSearchPath() );
   modloader->alwaysRecomp( true );
   modloader->saveModules( false );
   modloader->compileInMemory( opt_compmem );
   modloader->sourceEncoding( "utf-8" );

   int32 error;
   if ( opt_path == "" )
      opt_path = ".";
   else
      modloader->addDirectoryFront( opt_path );

   if ( opt_output != "" )
   {
      fs_out->create( opt_output, FileStream::e_aUserWrite | FileStream::e_aReadOnly );
      if( fs_out->bad() )
      {
         stdErr->writeString( "faltest: FATAL - can't open output file " + opt_output + "\n" );
         exit(1);
      }
      output = fs_out;
   }
   else {
      output = stdOut;
   }


   DirEntry *entry = Sys::fal_openDir( opt_path, error );
   if ( entry == 0 )
   {
      String err = "faltest: FATAL - can't open " + opt_path + " (error ";
      err.writeNumber( (int64) error );
      err += "\n";
      stdErr->writeString( err );
      exit(1);
   }

   {
      String filename;
      while( entry->read(filename) )
      {
         if ( filename.find( ".fal" ) == filename.length() - 4 )
         {
            FileStream script;
            // TODO: use the correct transcoder.
			   String temp = opt_path + "/" + filename;
            script.open( temp );
            if ( script.good() )
            {
               ScriptData *data = readProperties( filename, script );
               if ( data !=  0 )
               {
                  scriptMap[ data->id() ] = data;
                  String category;
                  if ( data->getProperty( "Category", category ) )
                     categoryMap[ category ].pushBack( data );
               }
            }
         }
      }

      Sys::fal_closeDir( entry );
   }


   filterScripts();
   if ( opt_justlist )
   {
      listScripts();
      delete stdOut;
      delete stdErr;
      return 0;
   }

   // ensure correct accounting by removing extra data.
   modloader->compiler().reset();

   // reset memory tests
   s_totalMem = gcmallocated();
   s_totalOutBlocks = memPool->allocatedItems();

   executeTests( modloader );

   // in context to have it destroyed on exit
   {
      output->writeString( "\n" );
      if( opt_verbose )
         output->writeString( "-----------------------------------------------------------------------\n" );

      String completed = "Completed ";
      completed.writeNumber( (int64) passedCount + failedCount );
      completed += " tests, passed ";
      completed.writeNumber( (int64) passedCount );
      completed += ",  failed ";
      completed.writeNumber( (int64) failedCount );
      completed += "\n";
      output->writeString( completed );

      stdOut->writeString( "\n" );
      if( opt_verbose && opt_output != "" )
      {
         stdOut->writeString( "-----------------------------------------------------------------------\n" );
         output->writeString( completed );
      }

      if( opt_timings ) {
         if ( opt_verbose ) {
            output->writeString( "Recorded timings: \n" );
            String temp = "Total compilation time: ";
            temp.writeNumber( total_time_generate, TIME_PRINT_FMT );
            temp += " (mean: ";
            temp.writeNumber( total_time_generate / (passedCount + failedCount), TIME_PRINT_FMT );
            temp += ")\n";

            temp += "Total generation time: ";
            temp.writeNumber( total_time_generate, TIME_PRINT_FMT );
            temp += " (mean: ";
            temp.writeNumber( total_time_generate / (passedCount + failedCount), TIME_PRINT_FMT );
            temp += ")\n";

            temp += "Total link time: ";
            temp.writeNumber( total_time_link, TIME_PRINT_FMT );
            temp += " (mean: ";
            temp.writeNumber( total_time_link / (passedCount + failedCount), TIME_PRINT_FMT );
            temp += ")\n";

            temp += "Total execution time: ";
            temp.writeNumber( total_time_execute, TIME_PRINT_FMT );
            temp += " (mean: ";
            temp.writeNumber( total_time_execute / (passedCount + failedCount), TIME_PRINT_FMT );
            temp += ")\n";

            double actTime = total_time_compile + total_time_generate + total_time_link + total_time_execute;
            temp += "Total activity time: ";
            temp.writeNumber( actTime, TIME_PRINT_FMT );
            temp += " (mean: ";
            temp.writeNumber( actTime / (passedCount + failedCount), TIME_PRINT_FMT );
            temp += ")\n";

            temp += "Total application time: ";
            temp.writeNumber( Sys::_seconds() - appTime, TIME_PRINT_FMT );
            temp += "\n";

            output->writeString( temp );
         }
         else {
            output->writeString( "Recorded timings: " );
            String temp;
            temp.writeNumber( total_time_compile, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( total_time_generate, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( total_time_link, TIME_PRINT_FMT );
            temp += " ";
            temp.writeNumber( total_time_execute, TIME_PRINT_FMT );
            temp += "\n";
            output->writeString( temp );

            double actTime = total_time_compile + total_time_generate + total_time_link + total_time_execute;
            temp = "Total time: ";
            temp.writeNumber( actTime, TIME_PRINT_FMT );
            temp += "\n";
            output->writeString( temp );
         }
      }
   }

   if ( opt_checkmem )
   {
      memPool->performGC();
      long blocks = memPool->allocatedItems() - s_totalOutBlocks;
      long mem = gcmallocated() - s_totalMem;

      String temp = "Final memory balance: ";
      temp = "Final memory balance: ";
      temp.writeNumber( (int64) mem );
      temp += "/";
      temp.writeNumber( (int64) blocks );
      temp += "\n";
      output->writeString( temp );
   }


   stdOut->writeString( "faltest: done.\n" );

   delete stdOut;
   delete stdErr;

   if ( failedCount > 0 )
       return 2;
   return 0;
}

/* end of faltest.cpp */
