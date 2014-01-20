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

#define MOD_FALCON_PROVIDER "mod_falcon for Apache 2.x"

#include <falcon/wopi/mem_sm.h>
#include <falcon/wopi/file_sm.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/scriptrunner.h>
#include <falcon/engine.h>
#include <falcon/vm.h>
#include <falcon/autocstring.h>

#include <stdio.h>

#include "mod_falcon.h"
#include "mod_falcon_config.h"
#include "apache_errhand.h"
#include "apache_stream.h"
#include "apache_request.h"
#include "apache_reply.h"
#include "apache_ll.h"

#include <http_log.h>
#include <util_filter.h>

using namespace Falcon;

/** Module wide VM */
static VMachine* s_vm = 0;
static ApacheLogListener* the_log_listener = 0;

//=============================================
// Global pointers to the core and RTL modules
//

//=============================================================
// Utilities
//

void falcon_mod_load_ini()
{
   Falcon::WOPI::Wopi* templateWopi = new Falcon::WOPI::Wopi;
   Falcon::WOPI::ErrorHandler* errorHand = new ApacheErrorHandler;

   the_falcon_config->errHand = errorHand;
   the_falcon_config->templateWopi = templateWopi;

   Falcon::String iniName(the_falcon_config->init_file);
    try
    {
       Falcon::Stream* stream = Falcon::Engine::instance()->vfs().openRO( iniName );
       Falcon::Transcoder* tc = Falcon::Engine::instance()->getTranscoder("utf8");
       fassert(tc != 0);
       Falcon::TextReader tr( stream, tc );
       Falcon::String errors;
       if( ! templateWopi->configFromIni( &tr, errors ) )
       {
          Falcon::AutoCString cs(errors);
          ap_log_perror( APLOG_MARK, APLOG_WARNING, 0, the_falcon_config->pool,
             "Error in Falcon configuration file \"%s\": %s",
             the_falcon_config->init_file, cs.c_str() );
       }

       errorHand->loadConfigFromWopi(templateWopi);
       templateWopi->configureAppLogListener( the_log_listener );

       the_falcon_config->loaded = 1;
    }
    catch( Falcon::Error *e )
    {
       Falcon::AutoCString cs(e->describe(true));
       the_falcon_config->loaded = -1;

       ap_log_perror( APLOG_MARK, APLOG_WARNING, 0, the_falcon_config->pool,
          "I/O error in accessing Falcon configuration file \"%s\": %s",
          the_falcon_config->init_file, cs.c_str() );
       e->decref();
    }
}


static void start_engine( apr_pool_t* p )
{
   Engine::init();
   s_vm = new VMachine;

   ApacheLogListener* ll = new ApacheLogListener(p);
   the_log_listener = ll;
   ll->facility(Falcon::Log::fac_all);
   ll->logLevel(Falcon::Log::lvl_debug2);
   Engine::instance()->log()->addListener(ll);

   falcon_mod_load_ini();
}

extern "C" {

static int falcon_handler(request_rec *request)
{
   String path;
   bool phand = false;

   // should we handle this as a special path?
   if ( strcmp( request->handler, FALCON_PROGRAM_HANDLER) == 0 )
   {
      phand = true;
   }
   else if ( strcmp( request->handler, FALCON_PROGRAM_FTD) == 0 )
   {
      phand = true;
   }
   else if( strcmp(request->handler, FALCON_FTD_TYPE) == 0 )
   {
      phand = true;
   }
   else if ( strcmp(request->handler, FALCON_FAM_TYPE) != 0 &&
        strcmp(request->handler, FALCON_FAL_TYPE) != 0 )
   {
      return DECLINED;
   }

   // verify that the file exists; decline otherwise
   const char *script_name = 0;
   falcon_dir_config* dircfg = (falcon_dir_config*)ap_get_module_config(request->per_dir_config, &falcon_module ) ;
   
   // Ok, we're here, but did we start our engine?
   if( s_vm == 0 )
   {
      start_engine(the_falcon_config->pool);
   }

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
   else if( the_falcon_config->loaded == -1 )
   {
      ap_log_perror( APLOG_MARK, APLOG_ERR, 0, request->pool,
         "Refused processing of \"%s\", because module failed to load configuration.",
         request->filename );
      return DECLINED;
   }
   else
   {
      script_name = request->filename;
      URI uri(script_name);
      FileStat::t_fileType st = Engine::instance()->vfs().fileType(uri, true);
      if ( st != FileStat::_normal  )
      {
         // sorry, the file do not exists, or we cannot access it.
         return DECLINED;
      }
   }

   WOPI::ErrorHandler* eh = static_cast<WOPI::ErrorHandler*>(the_falcon_config->errHand);
   WOPI::ScriptRunner runner(MOD_FALCON_PROVIDER, s_vm, Engine::instance()->log(), eh );

   WOPI::Wopi* wopi = static_cast<WOPI::Wopi*>(the_falcon_config->templateWopi);

   String error;
   String loadPath, srcEnc, textEnc;

   if( wopi->getConfigValue( WOPI::OPT_LoadPath, loadPath, error ) )
   {
      runner.loadPath(loadPath);
   }

   if( wopi->getConfigValue( WOPI::OPT_SourceEncoding, srcEnc, error ) )
   {
      runner.sourceEncoding(srcEnc);
   }

   if( wopi->getConfigValue( WOPI::OPT_OutputEncoding, textEnc, error ))
   {
      runner.textEncoding(textEnc);
   }

   Falcon::Engine::instance()->log()->log(Falcon::Log::fac_app,
                 Falcon::Log::lvl_info,
                 String("Running ").A(script_name));

   ApacheRequest* arequest = new ApacheRequest(0,request);
   ApacheReply* areply = new ApacheReply(0,request);
   ApacheStream* astream = new ApacheStream(request);
   WOPI::Client client( arequest, areply, astream );
   runner.run(&client, script_name, wopi );

   return OK;
}


//========================================
// Falcon module registration
//
void falcon_register_hook( apr_pool_t *p )
{
   // create falcon standard modules.
   ap_log_perror( APLOG_MARK, APLOG_INFO, 0, p, "-- registering module handler" );
   ap_hook_handler(falcon_handler, NULL, NULL, APR_HOOK_MIDDLE);
}

}

/* falcon_functions.cpp */
