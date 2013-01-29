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

#include "int_mode.h"

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
      if( m_options.log_file == "-" ){
         m_logger->m_logfile = new TextWriter(new StdOutStream(), true);
      }
      else if( m_options.log_file == "%" ) {
         m_logger->m_logfile = new TextWriter(new StdErrStream(), true);
      }
      else {
         try {
            Stream* out = Engine::instance()->vfs().create(
                     m_options.log_file,
                     VFSIface::CParams().append().wrOnly()
                     );
            m_logger->m_logfile = new TextWriter( out, true );
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
      else
      {
         if ( scriptPos <= 0 )
         {
            out.write( "Please, add a filename (for now)\n" );
            return;
         }

         String script = argv[scriptPos-1];
         log->log(Log::fac_app, Log::lvl_info, String("Starting execution of: ") + script );
         launch( script );
      }
   }
   catch( Error* e )
   {
      log->log( Log::fac_app, Log::lvl_critical, String("Terminating with error: ") + e->describe() );
      out.write( "falcon: Terminating with error\n" + e->describe() +"\n");
      e->decref();
   }

   log->removeListener( m_logger );
}


void FalconApp::interactive()
{
   IntMode intmode( this );
   intmode.run();
}


void FalconApp::launch( const String& script )
{
   Log* log = Engine::instance()->log();

   // Create the virtual machine -- that is used also for textual output.
   VMachine vm;
   vm.setStdEncoding( m_options.io_encoding );

   // Ok, we opened the file; prepare the space (and most important, the loader)
   ModSpace* ms = vm.modSpace();
   ModLoader* loader = ms->modLoader();
   loader->sourceEncoding( m_options.io_encoding );

   // do we have a load path?
   loader->setSearchPath(".");
   if( m_options.load_path.size() > 0 )
   {
      // Is the load path totally substituting?
      log->log(Log::fac_app, Log::lvl_detail, String("Setting load path to: ") + m_options.load_path );
      loader->addSearchPath(m_options.load_path);
   }

   if( ! m_options.ignore_syspath )
   {
      loader->addFalconPath();
   }
   else {
      log->log(Log::fac_app, Log::lvl_detail, String("Ignoring system paths") );
   }

   log->log(Log::fac_app, Log::lvl_info, String("System load path is: ") + loader->getSearchPath() );

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
   Process* loadProc = ms->loadModule( script, true, false, true );

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
         Process* process = vm.createProcess();
         process->mainContext()->call( mod->getMainFunction() );
         process->start();
         process->wait();

         m_exitValue = (int) process->result().forceInteger();

         log->log(Log::fac_app, Log::lvl_info, String("Script complete, exit value: ").N(m_exitValue) );
         //log->log(Log::fac_app, Log::lvl_detail, String("Exit value as item: ") + process->result().describe(3,128) );
      }
      else {
         log->log(Log::fac_app, Log::lvl_info, String("Main module hasn't a main function, terminating") );
      }
      mod->decref();
   }
   catch( ... )
   {
      mod->decref();
      throw;
   }
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
