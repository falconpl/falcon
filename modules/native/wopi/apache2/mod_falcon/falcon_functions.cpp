/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_functions.cpp

   Falcon module for Apache 2
   Plugin implementation file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/mem_sm.h>
#include <falcon/wopi/file_sm.h>
#include <falcon/wopi/wopi_ext.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/replystream.h>

#include <falcon/engine.h>
#include <falcon/dir_sys.h>
#include <falcon/time_sys.h>
#include <falcon/transcoding.h>

#include <stdio.h>

#include "mod_falcon.h"
#include "mod_falcon_config.h"
#include "mod_falcon_error.h"
#include "mod_falcon_vm.h"
#include "mod_falcon_watchdog.h"

#include "apache_errhand.h"
#include "apache_output.h"
#include "apache_stream.h"
#include "apache_request.h"
#include "apache_reply.h"

#include <http_log.h>
#include <util_filter.h>

#ifdef APLOG_USE_MODULE
   APLOG_USE_MODULE(falcon);
#endif

void ap_log_error_cb( const Falcon::String& msg, void* data )
{
   Falcon::AutoCString cmsg( msg );
   request_rec* request = (request_rec*) data;

   ap_log_rerror( APLOG_MARK, APLOG_INFO, 0, request,
      "%s", cmsg.c_str() );

}

//=============================================
// Global pointers to the core and RTL modules
//
static Falcon::Module *s_core =0;
static Falcon::Module *s_ext =0;
static Falcon::WOPI::SessionManager *s_session = 0;
static unsigned int s_entropy = 0;


// Startup this module
static void module_startup()
{
   Falcon::Engine::Init();
   Falcon::memPool->stop();
   //Falcon::memPool->rampMode( RAMP_MODE_STRICT_ID );

   // create also the core; we know it's empty.
   s_core = Falcon::core_module_init();

   // tell the VM to create our request and reply object
   s_ext = Falcon::WOPI::wopi_module_init( ApacheRequest::factory , ApacheReply::factory );

   if( the_falcon_config->cacheModules )
   {
      Falcon::Engine::cacheModules( true );
   }
         
   apr_pool_create(&falconWatchdogData.threadPool, NULL);      
   apr_status_t rv = apr_thread_create( 
         &falconWatchdogData.watchdogThread, 
         NULL, 
         &watchdog, 
         &falconWatchdogData, falconWatchdogData.threadPool );
  
   fprintf( stderr, "Falcon WOPI module for Apache2 %d(%s)\n", rv,
          rv == APR_SUCCESS ? "launched watchdog" : "failed to launch watchdog");

}

extern "C" {

static int falcon_handler(request_rec *request)
{
   Falcon::String path;
   bool phand = false;
   bool force_ftd = false;

   // should we handle this as a special path?
   if ( strcmp( request->handler, FALCON_PROGRAM_HANDLER) == 0 )
   {
      phand = true;
   }
   else if ( strcmp( request->handler, FALCON_PROGRAM_FTD) == 0 )
   {
      phand = true;
      force_ftd = true;
   }
   else if ( strcmp(request->handler, FALCON_FAM_TYPE) != 0 &&
        strcmp(request->handler, FALCON_FAL_TYPE) != 0 &&
        strcmp(request->handler, FALCON_FTD_TYPE) != 0)
   {
      return DECLINED;
   }

   // first time in this process?
   if( s_core == 0 )
   {
      module_startup();
   }

   Falcon::numeric startedAt = Falcon::Sys::Time::seconds();

   // verify that the file exists; decline otherwise
   const char *script_name = 0;
   const char *load_path = 0;
   falcon_dir_config* dircfg = (falcon_dir_config*)ap_get_module_config(request->per_dir_config, &falcon_module ) ;
   
   if ( phand )
   {
      // do we have a request configuration?
      if( dircfg->falconHandler[0] != '\0' )
      {
         script_name = dircfg->falconHandler;
      }
      else if( the_falcon_config->loaded != -1 )
      {
         script_name = the_falcon_config->falconHandler;
      }

      // else, go to the next the_falcon_config->loaded != -1 check
   }
   else
   {
      script_name = request->filename;
      Falcon::FileStat::e_fileType st;
      if ( ! Falcon::Sys::fal_fileType( script_name, st ) )
      {
         // sorry, the file do not exists, or we cannot access it.
         return DECLINED;
      }
   }

  
   // we accepted. but are we working?
   if( the_falcon_config->loaded == -1 )
   {
      falcon_mod_write_errorstring( request, "<HTML>\n<BODY>\n<H1>FALCON module error</H1>\n"
         "<p>Couldn't load Falcon configuration file. See the error log of this server "
         "for further details.</p>\n"
         "</BODY>\n"
         "</HTML>\n"
         );
      return OK;
   }

   // Set the load path
   if( dircfg->loadPath[0] != '\0' )
   {
      load_path = dircfg->loadPath;
   }
   else if( the_falcon_config->loaded != -1 )
   {
      load_path = the_falcon_config->loadPath;
   }

 
   // prepare the session manager for this process.
   if ( s_session == 0 )
   {
      if ( the_falcon_config->sessionMode == FM_DEFAULT_SESSION_MODE )
      {
         s_session = new Falcon::WOPI::FileSessionManager( the_falcon_config->uploadDir );
      }
      else
      {
         s_session = new Falcon::WOPI::MemSessionManager;
      }

      s_session->timeout( the_falcon_config->sessionTimeout );
      s_session->startup();
   }

   // we need some random data
   // for sure, the request address is unique within a time constraint.
   s_entropy += (((long)(startedAt * 1000))%0xFFFFFFFF + ((long) request));
   srand( s_entropy );

   //it's a Falcon module.
   ap_log_rerror( APLOG_MARK, APLOG_INFO, 0, request,
      "Accepting a request from falcon %s", request->filename );

   // first of all, we need a module loader to load the script.
   // The parameter is the search path for where to search our module
   Falcon::ModuleLoader theLoader;

   // As we want to use standard Falcon installation,
   // tell the loader that is safe to search module in system path
   if( load_path != 0 && load_path[0] != '\0' )
   {
      theLoader.setSearchPath( load_path );
   }
   else
   {
      theLoader.setSearchPath( "" );
      theLoader.addFalconPath();
   }

   // also, instruct the module loader to take the script path as added search path
   Falcon::Path scriptPath( request->canonical_filename );

   if ( scriptPath.isValid() )
   {
      Falcon::String spath;
      scriptPath.getLocation( spath );
      theLoader.addDirectoryFront( spath );
   }

   // broadcast the same path to the application.
   Falcon::Engine::setSearchPath( theLoader.getSearchPath() );
   
   path = theLoader.getSearchPath();
   // for now, let's say we're utf-8
   theLoader.sourceEncoding( "utf-8" );
   Falcon::Engine::setEncodings( "utf-8", "utf-8" );

   if( force_ftd )
      theLoader.compileTemplate( true );

   // Now we need our error handler to manage possible problems in the next steps.
   // First, create the output brigade we'll need to deal with output.
   ApacheOutput aoutput( request );

   // then create an error handler that may use this output structure.
   ApacheErrorHandler errhand( the_falcon_config->errorMode, &aoutput );

   // Allow the script to load iteratively other resources it may need.
   ApacheRequest* ar = 0;
   Falcon::uint32 nSessionToken = 0;

   // We are ready to go. Let's create our VM and link in minimal stuff
   // we'll handle errors here
   ApacheVMachine* vm = new ApacheVMachine();

   // perform link of standard modules.
   vm->link( s_core );  // add the core module
   vm->link( s_ext );

   // prepare the request class.
   Falcon::Item* i_req = vm->findGlobalItem( "Request" );
   fassert( i_req != 0 );
   Falcon::Item* i_upld = vm->findGlobalItem( "Uploaded" );
   fassert( i_upld != 0 );
   Falcon::Item* i_reply = vm->findGlobalItem( "Reply" );
   fassert( i_reply != 0 );
   Falcon::Item* i_wopi = vm->findGlobalItem( "Wopi" );
   fassert( i_wopi != 0 );

   // Configure the Apache Request
   ar = Falcon::dyncast<ApacheRequest*>( i_req->asObject() );

   // And the reply here
   ApacheReply* reply = Falcon::dyncast<ApacheReply*>( i_reply->asObject() );
   reply->init( request, &aoutput );

   // init and configure
   ar->init( request, i_upld->asClass(), reply, s_session );
   ar->base()->startedAt( startedAt );
   // record the session token for automatic close at end.
   nSessionToken = ar->base()->sessionToken();

   vm->stdOut( new Falcon::WOPI::ReplyStream( reply ) );
   vm->stdErr( new Falcon::WOPI::ReplyStream( reply ) );

   // sets the options
   ar->base()->setMaxMemUpload( the_falcon_config->maxMemUpload );
   if( the_falcon_config->uploadDir[0] != 0 )
      ar->base()->setUploadPath( the_falcon_config->uploadDir );
   ar->configFromModule( vm->mainModule()->module() );

   // configure the WOPI persistent object
   if ( the_falcon_config->pdataDir[0] != 0 )
      Falcon::dyncast<Falcon::WOPI::CoreWopi*>(i_wopi->asObject())->wopi()->dataLocation( the_falcon_config->pdataDir );
   Falcon::dyncast<Falcon::WOPI::CoreWopi*>(i_wopi->asObject())->configFromModule( vm->mainModule()->module() );


   // ok start processing of the input request.
   ar->process();

   // ok; now prepare the scriptName and scriptPath variables, and set the current working directory
   // so that the script is ready.
   if ( scriptPath.isValid() )
   {
      Falcon::CoreString *spath = new Falcon::CoreString;
      Falcon::CoreString *sfile = new Falcon::CoreString;

      scriptPath.getLocation( *spath );
      scriptPath.getFilename( *sfile );
      Falcon::Item *i_scriptPath = vm->findGlobalItem( "scriptPath" );
      fassert( i_scriptPath != 0 );
      *i_scriptPath = spath;
      Falcon::Item *i_scriptName = vm->findGlobalItem( "scriptName" );
      fassert( i_scriptName != 0 );
      *i_scriptName = sfile;

      int fs;
      Falcon::Sys::fal_chdir( *spath, fs );
   }
   
   FALCON_WATCHDOG_TOKEN* wdt = 0;
   try {
      Falcon::Runtime rt( &theLoader );
      rt.loadFile( script_name );

      // tell the watchdog we're going to 
      if( the_falcon_config->runTimeout > 0 )
      {
         // approximate good size.
         int cbloops = the_falcon_config->runTimeout * 1000;
         vm->callbackLoops( cbloops > 10000 ? 10000 : cbloops );

         wdt = watchdog_push_vm( request, vm, the_falcon_config->runTimeout, reply );
      }
      // try to link our module and its dependencies.
      // -- It may fail if there are some undefined symbols
      vm->link( &rt );
      vm->launch();

      // close our stream
      aoutput.close();
   }
   catch( Falcon::Error* error )
   {
      reply->commit();
      errhand.handleError( error );
      aoutput.close();
   }
   
   if ( wdt != 0 ) {
      watchdog_vm_is_done( wdt );
   }

   // release all our sessions
   s_session->releaseSessions( nSessionToken );

   if( vm != 0 )
   {      
      void *temp_files = ar->base()->getTempFiles();
      vm->finalize();
      Falcon::memPool->start();
      //Falcon::memPool->performGC();

      if ( temp_files != 0 )
         Falcon::WOPI::Request::removeTempFiles( temp_files, request, ap_log_error_cb );
   }

   return OK;
}


//========================================
// Falcon module registration
//
void falcon_register_hook( apr_pool_t *mp )
{
   apr_status_t rv;
   
   fprintf( stderr, "Falcon " VERSION " -- WOPI module for Apache2 startup.");
   
   // Inititialize watchdog data
   falconWatchdogData.terminate = 0;
   falconWatchdogData.incomingVM = 0;
   falconWatchdogData.lastIncomingVM = 0;
   
   apr_thread_mutex_create(&falconWatchdogData.mutex, APR_THREAD_MUTEX_UNNESTED, mp);
   apr_thread_cond_create(&falconWatchdogData.cond, mp);
   
   
   
   // Launch the watchdog thread
   
   
   // create falcon standard modules.
   ap_hook_handler(falcon_handler, NULL, NULL, APR_HOOK_MIDDLE);
}

}

/* falcon_functions.cpp */
