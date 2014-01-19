/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_config.h

   Falcon specific configuration for the module falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <httpd.h>
#include <http_log.h>
#include <apr_strings.h>

#include <falcon/string.h>
#include <falcon/autocstring.h>
#include <falcon/stream.h>
#include <falcon/textreader.h>
#include <falcon/engine.h>
#include <falcon/vfsiface.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/errorhandler.h>

#include "mod_falcon_config.h"
#include "apache_errhand.h"

extern module AP_MODULE_DECLARE_DATA falcon_module;

falcon_mod_config *the_falcon_config;

//=============================================================
// Utilities
//

static int falcon_mod_load_ini( const char* fname, void* tw, void* eh, apr_pool_t* pool )
{
   Falcon::WOPI::Wopi& templateWopi = *static_cast<Falcon::WOPI::Wopi*>(tw);
   Falcon::WOPI::ErrorHandler& errorHand = *static_cast<Falcon::WOPI::ErrorHandler*>(eh);

   Falcon::String iniName(fname);
    try
    {
       Falcon::Stream* stream = Falcon::Engine::instance()->vfs().openRO( iniName );
       Falcon::Transcoder* tc = Falcon::Engine::instance()->getTranscoder("utf8");
       fassert(tc != 0);
       Falcon::TextReader tr( stream, tc );
       Falcon::String errors;
       if( ! templateWopi.configFromIni( &tr, errors ) )
       {
          Falcon::AutoCString cs(errors);
          ap_log_perror( APLOG_MARK, APLOG_WARNING, 0, pool,
             "Error in Falcon configuration file \"%s\": %s",
             fname, cs.c_str() );
       }

       errorHand.loadConfigFromWopi(&templateWopi);
    }
    catch( Falcon::Error *e )
    {
       Falcon::AutoCString cs(e->describe(true));

       ap_log_perror( APLOG_MARK, APLOG_WARNING, 0, pool,
          "I/O error in accessing Falcon configuration file \"%s\": %s",
          fname, cs.c_str() );
       e->decref();

       return -1;
    }

    return 1;
}

//=============================================================
// APACHE hooks
//
void *falcon_mod_create_config(apr_pool_t *p, server_rec *)
{
   falcon_mod_config *newcfg;
   newcfg = (falcon_mod_config *) apr_pcalloc(p, sizeof(falcon_mod_config));
   apr_cpystrn( newcfg->init_file, DEFAULT_CONFIG_FILE, MAX_LOAD_PATH);
   newcfg->pool = p;
   newcfg->loaded = 0;
   newcfg->templateWopi = new Falcon::WOPI::Wopi;
   newcfg->errHand = new ApacheErrorHandler;

   apr_cpystrn( newcfg->falconHandler, DEFAULT_HANDLER_NAME, MAX_LOAD_PATH );
   apr_cpystrn( newcfg->init_file, "", MAX_LOAD_PATH );

   // initialize the global instance of our configuration
   the_falcon_config = newcfg;

   return (void *) newcfg;
}

const char *falcon_mod_set_config(cmd_parms *parms, void *, const char *arg)
{
   falcon_mod_config *cfg = (falcon_mod_config *)
                              ap_get_module_config( parms->server->module_config, &falcon_module);
   
   apr_cpystrn( cfg->init_file, arg, MAX_LOAD_PATH );
   
   // perform loading
   if ( ! falcon_mod_load_config( the_falcon_config ) )
      cfg->loaded = -1;
   else
      cfg->loaded = 1;
   return NULL;
}

//===================================================================
// Local directory level config.


void *falcon_mod_create_dir_config(apr_pool_t *p, char *)
{
   falcon_dir_config *newcfg;
   newcfg = (falcon_dir_config *) apr_pcalloc(p, sizeof(falcon_dir_config));
   newcfg->falconHandler[0] = '\0';
   newcfg->init_file[0] = '\0';
   newcfg->templateWopi = new Falcon::WOPI::Wopi;
   newcfg->errHand = new ApacheErrorHandler;
   
   return (void *) newcfg;
}


void *falcon_mod_merge_dir_config(apr_pool_t *p, void* BASE, void* ADD )
{
   falcon_dir_config *newcfg;
   newcfg = (falcon_dir_config *) apr_pcalloc(p, sizeof(falcon_dir_config));
   newcfg->falconHandler[0] = '\0';
   newcfg->init_file[0] = '\0';

   falcon_dir_config *base = (falcon_dir_config *) BASE;
   falcon_dir_config *add = (falcon_dir_config *) ADD;

   if ( add->falconHandler[0] != '\0' )
   {
      apr_cpystrn( newcfg->falconHandler, add->falconHandler, MAX_LOAD_PATH );
   }
   else if( base->falconHandler[0] != '\0' )
   {
      apr_cpystrn( newcfg->falconHandler, base->falconHandler, MAX_LOAD_PATH );
   }

   if ( add->init_file[0] != '\0' )
   {
      apr_cpystrn( newcfg->init_file, add->init_file, MAX_LOAD_PATH );
      falcon_mod_load_ini( newcfg->init_file, newcfg->templateWopi, newcfg->errHand, p );
   }
   else if( base->init_file[0] != '\0' )
   {
      apr_cpystrn( newcfg->init_file, base->init_file, MAX_LOAD_PATH );

      Falcon::WOPI::Wopi* basewopi = static_cast<Falcon::WOPI::Wopi*>(base->templateWopi);
      Falcon::WOPI::Wopi* newwopi = static_cast<Falcon::WOPI::Wopi*>(newcfg->templateWopi);
      newwopi->configFromWopi( *basewopi );

      Falcon::WOPI::ErrorHandler* eh = static_cast<Falcon::WOPI::ErrorHandler*>(newcfg->errHand);
      eh->loadConfigFromWopi(newwopi);
   }
   
   return (void *) newcfg;
}


const char *falcon_mod_set_handler(cmd_parms *parms, void *DIRCFG, const char *arg)
{
   if ( DIRCFG != 0 )
   {
      falcon_dir_config *dircfg = (falcon_dir_config *) DIRCFG;
      apr_cpystrn( dircfg->falconHandler, arg, MAX_LOAD_PATH );
   }
   else {
      // Server wide configuration?
      falcon_mod_config* modconfig = (falcon_mod_config*)ap_get_module_config(parms->server, &falcon_module );
      apr_cpystrn( modconfig->falconHandler, arg, MAX_LOAD_PATH );
   }
   
   return NULL;
}


//====================================================================
// Config loader implementation.
//
int falcon_mod_load_config( falcon_mod_config *cfg )
{
   ap_log_perror( APLOG_MARK, APLOG_INFO, 0, cfg->pool,
      "Performing lazy initialization of Falcon Config file %s",
      cfg->init_file );

   return falcon_mod_load_ini( cfg->init_file, cfg->templateWopi, cfg->errHand, cfg->pool );
}
