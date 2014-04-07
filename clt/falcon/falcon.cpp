/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.cpp

   Falcon command line
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/falcon.h>
#include <falcon/modloader.h>
#include <falcon/trace.h>
#include <falcon/stringstream.h>

#include <falcon/itemarray.h>    // for args
#include <falcon/gclock.h>       // for args

#include "int_mode.h"
#include "testmode.h"

#include <memory>

using namespace Falcon;

//==============================================================
// The application
//
FalconApp::FalconApp():
   m_exitValue(0)
{
   m_logger = new Logger;
}

FalconApp::~FalconApp()
{
   m_logger->decref();
}

void FalconApp::guardAndGo( int argc, char* argv[] )
{
   int scriptPos = 0;
   m_options.parse( argc, argv, scriptPos );
   if( m_options.m_justinfo )
   {
      return;
   }

   Log* log = Engine::instance()->log();

   if( m_options.log_level >= 0 )
   {
      Stream* out;
      if( m_options.log_file == "-" ){
         out = new StdOutStream();
         m_logger->m_logfile = new TextWriter( out );
         out->decref();
      }
      else if( m_options.log_file == "%" ) {
         out = new StdErrStream();
         m_logger->m_logfile = new TextWriter( out );
         out->decref();
      }
      else {
         try {
            out = Engine::instance()->vfs().create(
                     m_options.log_file,
                     VFSIface::CParams().append().wrOnly()
                     );
            m_logger->m_logfile = new TextWriter( out  );
            out->decref();
         }
         catch( Error* err )
         {
            Engine::die(String("Cannot create requried log file: ") + err->describe() );
         }
      }

      log->addListener( m_logger );
   }

   // prepare the trace file
#ifndef NDEBUG
   if( m_options.trace_file != "" )
   {
      TRACE_ON_FILE( m_options.trace_file.c_ize() );
   }
#endif

   TextWriter out(new StdOutStream);
   try
   {
      if( m_options.interactive )
      {
         log->log(Log::fac_app, Log::lvl_info, String("Going interactive") );
         interactive();
      }
      else if( m_options.testMode )
      {
         log->log(Log::fac_app, Log::lvl_info, String("Going testmode") );
         testMode();
      }
      else if( m_options.m_bEval )
      {
         evaluate( m_options.m_sEval, false );
      }
      else
      {
         if ( scriptPos <= 0 )
         {
            log->log(Log::fac_app, Log::lvl_info, "Reading script from standard input" );
            evaluate( "", true );
         }
         else {
            String script = argv[scriptPos-1];
            log->log(Log::fac_app, Log::lvl_info, String("Starting execution of: ") + script );
            launch( script, argc, argv, scriptPos );
         }
      }
   }
   catch( Error* e )
   {
      String errDesc;
      bool bAddSign, bAddPath, bAddParams;
      m_options.getErrorReportMode( bAddPath, bAddParams, bAddSign );
      e->describeTo(errDesc, bAddPath, bAddParams, bAddSign );

      log->log( Log::fac_app, Log::lvl_critical, String("Terminating with error: ") + errDesc );
      // unpacking errors
      if( e->errorCode() == e_compile || e->errorCode() == e_link_error )
      {
         String temp;
         e->describeSubErrors(temp,false);
         out.write( "falcon: Terminating with error\n" + temp + "\n");
      }
      else
      {
         out.write( "falcon: Terminating with error\n" + errDesc +"\n");
      }
      e->decref();
   }

   log->removeListener( m_logger );
}


void FalconApp::interactive()
{
   IntMode intmode( this );
   intmode.run();
}


void FalconApp::testMode()
{
   TestMode tm(this);
   tm.perform();
}


void FalconApp::launch( const String& script, int argc, char* argv[], int pos )
{
   Log* log = Engine::instance()->log();

   // Create the virtual machine -- that is used also for textual output.
   VMachine& vm = *(new VMachine);
   Process* process = vm.createProcess();
   configureVM( vm, process );

   ModSpace* ms = process->modSpace();

   // by now we shoild have args around.
   {
      Item* args = ms->findExportedValue("args");
      fassert( args != 0 );
      ItemArray* arrArgs = new ItemArray(argc-pos);
      GCLock* lock = Engine::collector()->lock(Item(arrArgs->handler(), arrArgs));
      args->setUser(FALCON_GC_HANDLE(arrArgs));
      for( int i = pos; i< argc; ++i )
      {
         String* str = new String;
         // todo, use the correct transcoder
         if( ! str->fromUTF8(argv[i]) ) {
            str->bufferize(argv[i]);
         }
         arrArgs->append(FALCON_GC_HANDLE(str));
      }

      lock->dispose();
   }

   // add the script path to the load path
   try
   {
      URI scriptUri( script );
      if( ! scriptUri.path().fulloc().empty() || ! scriptUri.scheme().empty() )
      {
         scriptUri.path().file("");
         scriptUri.fragment("");
         scriptUri.query() = "";
         ms->modLoader()->addSearchPath(scriptUri.encode());
      }
   }
   catch(Error *e)
   {
      log->log( Log::fac_app, Log::lvl_error, String( "Invalid script path: ") + e->describe(true) );
      // try anyhow to proceed.
      e->decref();
   }

   Process* loadProc = ms->loadModule( script, true, true, true );

   log->log(Log::fac_app, Log::lvl_info, String("Starting loader process on: ") + script );
   loadProc->start();
   loadProc->wait();
   log->log(Log::fac_app, Log::lvl_info, String("Main script load complete") );

   // get the main module
   Module* mod = static_cast<Module*>(loadProc->result().asInst());
   loadProc->decref();

   try {
      if( mod->getMainFunction() != 0 )
      {
         log->log(Log::fac_app, Log::lvl_info, String("Launching main script function") );

         process->mainContext()->call( mod->getMainFunction() );
         if( m_options.debug )
         {
            log->log(Log::fac_app, Log::lvl_info, String("Activating debug") );
            process->mainContext()->setBreakpointEvent();
         }
         process->start();
         process->wait();

         m_exitValue = (int) process->result().forceInteger();

         log->log(Log::fac_app, Log::lvl_info, String("Script complete, exit value: ").N(m_exitValue) );
         log->log(Log::fac_app, Log::lvl_detail, String("Exit value as item: ") + process->result().describe(3,128) );
      }
      else {
         log->log(Log::fac_app, Log::lvl_info, String("Main module hasn't a main function, terminating") );
      }

      log->log(Log::fac_app, Log::lvl_info, "Performing cleanup sequence" );
      mod->decref();
      process->decref();
      delete& vm;
      Engine::instance()->collector()->performGC(true);
      log->log(Log::fac_app, Log::lvl_info, "Performing cleanup sequence complete" );
   }
   catch( ... )
   {
      mod->decref();
      process->decref();
      delete& vm;
      throw;
   }
}



void FalconApp::evaluate( const String& script, bool useStdin )
{
   Log* log = Engine::instance()->log();

   // Create the virtual machine -- that is used also for textual output.
   VMachine& vm = *(new VMachine);
   Process* process = vm.createProcess();
   configureVM( vm, process );

   IntCompiler ic;
   Stream* input;
   if( useStdin )
   {
      input = new StdInStream;
      ic.sp().interactive(false);
   }
   else
   {
      input = new StringStream( script );
   }
   TextReader* tr = new TextReader(input);
   input->decref();
   Module* mod = 0;

   try {
      Module* mod = ic.compile(tr, "eval://", "<evaluated>", false);
      if( mod == 0 )
      {
         throw ic.makeError();
      }

      process->modSpace()->add(mod);

      // run main function
      Function* mainfunc = mod->getMainFunction();
      if( mainfunc != 0 )
      {
         log->log(Log::fac_app, Log::lvl_info, String("Launching evaluation on command line") );
         log->log(Log::fac_app, Log::lvl_detail, String("Evaluated script:\n") + script );
         process->start( mainfunc, 0, 0 );
         process->wait();
         m_exitValue = (int) process->result().forceInteger();

         log->log(Log::fac_app, Log::lvl_info, String("Script complete, exit value: ").N(m_exitValue) );
         log->log(Log::fac_app, Log::lvl_detail, String("Exit value as item: ") + process->result().describe(3,128) );
      }

      log->log(Log::fac_app, Log::lvl_info, "Performing cleanup sequence" );
      mod->decref();
      process->decref();
      delete& vm;
      delete tr;
      Engine::instance()->collector()->performGC(true);
      log->log(Log::fac_app, Log::lvl_info, "Performing cleanup sequence complete" );
   }
   catch( ... )
   {
      if (mod != 0) mod->decref();
      process->decref();
      delete& vm;
      delete tr;
      throw;
   }
}


void FalconApp::configureVM( VMachine& vm, Process* prc, Log* log )
{
   prc->setBreakCallback(&m_dbg);
   vm.setProcessorCount( m_options.num_processors );
   vm.setStdEncoding( m_options.io_encoding );
   if( ! prc->setStdEncoding( m_options.io_encoding ) )
   {
      throw FALCON_SIGN_XERROR(EncodingError, e_unknown_encoding, .extra(m_options.io_encoding));
   }

   // Ok, we opened the file; prepare the space (and most important, the loader)
   ModSpace* ms = prc->modSpace();
   ModLoader* loader = ms->modLoader();
   if ( ! loader->sourceEncoding( m_options.source_encoding ) )
   {
      throw FALCON_SIGN_XERROR(EncodingError, e_unknown_encoding, .extra(m_options.source_encoding));
   }

   // do we have a load path?
   loader->setSearchPath(".");
   if( m_options.load_path.size() > 0 )
   {
      // Is the load path totally substituting?
      if ( log ) log->log(Log::fac_app, Log::lvl_detail, String("Setting load path to: ") + m_options.load_path );
      loader->addSearchPath(m_options.load_path);
   }

   if( ! m_options.ignore_syspath )
   {
      loader->addFalconPath();
   }
   else {
      if ( log ) log->log(Log::fac_app, Log::lvl_detail, String("Ignoring system paths") );
   }

   if ( log ) log->log(Log::fac_app, Log::lvl_info, String("System load path is: ") + loader->getSearchPath() );

   // Now configure other options of the lodaer.
   // -- How to treat sources?
   if( m_options.ignore_sources )
   {
      loader->useSources( ModLoader::e_us_never );
   }
   else {
      if( m_options.force_recomp )
      {
         loader->useSources( ModLoader::e_us_always );
      }

      if( m_options.parse_ftd ) {
         loader->checkFTD( ModLoader::e_ftd_force );
      }
   }

   // -- Save modules?
   if( m_options.save_modules )
   {
      if( m_options.force_recomp )
      {
         loader->savePC( ModLoader::e_save_mandatory );
      }
      else
      {
         loader->savePC( ModLoader::e_save_try );
      }
   }
   else
   {
      loader->savePC( ModLoader::e_save_no );
   }

   ms->add( Engine::instance()->getCore() );

}

//===============================================================
// Logger
//

FalconApp::Logger::Logger():
         m_logfile(0)
{
}

FalconApp::Logger::~Logger()
{
   delete m_logfile;
}

void FalconApp::Logger::onMessage( int fac, int lvl, const String& message )
{
   String tgt;
   Log::formatLog(fac, lvl, message, tgt );
   m_logfile->writeLine( tgt );
   m_logfile->flush();
}


int main( int argc, char* argv[] )
{
   FalconApp app;
   app.guardAndGo( argc, argv );

   return app.m_exitValue;
}

/* end of falcon.cpp */
