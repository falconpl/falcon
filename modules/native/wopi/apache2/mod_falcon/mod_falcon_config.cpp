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

#include "mod_falcon_config.h"

extern module AP_MODULE_DECLARE_DATA falcon_module;

falcon_mod_config *the_falcon_config;

#ifdef APLOG_USE_MODULE
   APLOG_USE_MODULE(falcon);
#endif

//=============================================================
// APACHE hooks
//
void *falcon_mod_create_config(apr_pool_t *p, server_rec *s)
{
   falcon_mod_config *newcfg;
   newcfg = (falcon_mod_config *) apr_pcalloc(p, sizeof(falcon_mod_config));
   newcfg->init_file = DEFAULT_CONFIG_FILE;
   newcfg->pool = p;
   newcfg->loaded = 0;
   newcfg->errorMode = FM_ERROR_MODE_KIND;
   newcfg->maxUpload = FM_DEFAULT_MAX_UPLOAD*1024;
   newcfg->maxMemUpload = 0;
   newcfg->sessionTimeout = FM_DEFAULT_SESSION_TO;
   newcfg->sessionMode = FM_DEFAULT_SESSION_MODE;
   newcfg->cacheModules = 0;
   newcfg->runTimeout = FM_DEFAULT_RUN_TIMEOUT;

   apr_cpystrn( newcfg->uploadDir, FM_DEFAULT_UPDIR, MAX_UPLOAD_DIR );
   apr_cpystrn( newcfg->loadPath, "", MAX_UPLOAD_DIR );
   apr_cpystrn( newcfg->falconHandler, DEFAULT_HANDLER_NAME, MAX_UPLOAD_DIR );
   apr_cpystrn( newcfg->pdataDir, "", MAX_UPLOAD_DIR );

   // initialize the global instance of our configuration
   the_falcon_config = newcfg;

   return (void *) newcfg;
}

const char *falcon_mod_set_config(cmd_parms *parms, void *mconfig, const char *arg)
{
   falcon_mod_config *cfg = (falcon_mod_config *)
   ap_get_module_config( parms->server->module_config, &falcon_module);
   
   cfg->init_file = (char *) arg;
   
   // perform loading
   if ( ! falcon_mod_load_config( the_falcon_config ) )
      cfg->loaded = -1;
   else
      cfg->loaded = 1;
   return NULL;
}

const char *falcon_mod_set_cacheModules(cmd_parms *parms, void *mconfig, const char *arg)
{
   falcon_mod_config *cfg = (falcon_mod_config *)
   ap_get_module_config( parms->server->module_config, &falcon_module);

   if (apr_strnatcasecmp( arg, "on" ) == 0 )
   {
      cfg->cacheModules = 1;
   }
   else if (apr_strnatcasecmp( arg, "off" ) == 0 )
   {
      cfg->cacheModules = 0;
   }
   else
      return "FalconCacheModules must be \"on\" or \"off\"";

   return NULL;
}


const char *falcon_mod_set_runTimeout(cmd_parms *parms, void *mconfig, const char *arg)
{
   int timeout = atoi( arg );
   falcon_mod_config *cfg = (falcon_mod_config *)
   ap_get_module_config( parms->server->module_config, &falcon_module);

   
   if (timeout == 0 && arg[0] != '\0' )
   {
      return "FalconRunTimeout is a number of seconds (0=infinite)";
   }
   else
   {
      cfg->runTimeout = timeout;
   }

   return NULL;
}

//===================================================================
// Local directory level config.


void *falcon_mod_create_dir_config(apr_pool_t *p, char *s)
{
   falcon_dir_config *newcfg;
   newcfg = (falcon_dir_config *) apr_pcalloc(p, sizeof(falcon_dir_config));
   newcfg->falconHandler[0] = '\0';
   newcfg->loadPath[0] = '\0';
   newcfg->pdataDir[0] = '\0';
   
   return (void *) newcfg;
}

void *falcon_mod_merge_dir_config(apr_pool_t *p, void* BASE, void* ADD )
{
   falcon_dir_config *newcfg;
   newcfg = (falcon_dir_config *) apr_pcalloc(p, sizeof(falcon_dir_config));
   newcfg->falconHandler[0] = '\0';

   falcon_dir_config *base = (falcon_dir_config *) BASE;
   falcon_dir_config *add = (falcon_dir_config *) ADD;

   if ( add->falconHandler[0] != '\0' )
   {
      apr_cpystrn( newcfg->falconHandler, add->falconHandler, MAX_UPLOAD_DIR );
   }
   else if( base->falconHandler[0] != '\0' )
   {
      apr_cpystrn( newcfg->falconHandler, base->falconHandler, MAX_UPLOAD_DIR );
   }

   if ( add->loadPath[0] != '\0' )
   {
      apr_cpystrn( newcfg->loadPath, add->loadPath, MAX_UPLOAD_DIR );
   }
   else if( base->loadPath[0] != '\0' )
   {
      apr_cpystrn( newcfg->loadPath, base->loadPath, MAX_UPLOAD_DIR );
   }

   if ( add->pdataDir[0] != '\0' )
   {
      apr_cpystrn( newcfg->pdataDir, add->pdataDir, MAX_UPLOAD_DIR );
   }
   else if( base->pdataDir[0] != '\0' )
   {
      apr_cpystrn( newcfg->pdataDir, base->pdataDir, MAX_UPLOAD_DIR );
   }
   
   return (void *) newcfg;
}

const char *falcon_mod_set_handler(cmd_parms *parms, void *DIRCFG, const char *arg)
{
   if ( DIRCFG != 0 )
   {
      falcon_dir_config *dircfg = (falcon_dir_config *) DIRCFG;
      apr_cpystrn( dircfg->falconHandler, arg, MAX_UPLOAD_DIR );
   }
   else {
      // Server wide configuration?
      falcon_mod_config* modconfig = (falcon_mod_config*)ap_get_module_config(parms->server, &falcon_module );
      apr_cpystrn( modconfig->falconHandler, arg, MAX_UPLOAD_DIR );
   }
   
   return NULL;
}

const char *falcon_mod_set_path(cmd_parms *parms, void *DIRCFG, const char *arg)
{
   if ( DIRCFG != 0 )
   {
      falcon_dir_config *dircfg = (falcon_dir_config *) DIRCFG;
      apr_cpystrn( dircfg->loadPath, arg, MAX_UPLOAD_DIR );
   }
   else {
      // Server wide configuration?
      falcon_mod_config* modconfig = (falcon_mod_config*)ap_get_module_config(parms->server, &falcon_module );
      apr_cpystrn( modconfig->loadPath, arg, MAX_UPLOAD_DIR );
   }
   
   return NULL;
}



const char *falcon_mod_set_pdataDir(cmd_parms *parms, void *DIRCFG, const char *arg)
{
   if ( DIRCFG != 0 )
   {
      falcon_dir_config *dircfg = (falcon_dir_config *) DIRCFG;
      apr_cpystrn( dircfg->pdataDir, arg, MAX_UPLOAD_DIR );
   }
   else {
      // Server wide configuration?
      falcon_mod_config* modconfig = (falcon_mod_config*)ap_get_module_config(parms->server, &falcon_module );
      apr_cpystrn( modconfig->pdataDir, arg, MAX_UPLOAD_DIR );
   }

   return NULL;
}

//====================================================================
// Config loader implementation.
//
// We're having a VERY simple config file with just KEY = VALUE pairs.
//
int falcon_mod_load_config( falcon_mod_config *cfg )
{
   char buffer[256];

   ap_log_perror( APLOG_MARK, APLOG_INFO, 0, cfg->pool,
      "Performing lazy initialization of Falcon Config file %s",
      cfg->init_file );

   apr_file_t *fd;
   apr_status_t result = apr_file_open( &fd, cfg->init_file,
         APR_READ | APR_BUFFERED,
         APR_OS_DEFAULT,
         cfg->pool );

   if ( result != 0 )
   {
      ap_log_perror( APLOG_MARK, APLOG_ERR, result, cfg->pool,
         "Can't load Falcon initialization file %s",
         cfg->init_file );
      return 0;
   }

   // Try to load the file.
   result = apr_file_gets( buffer, 255, fd );
   int line = 1;
   while ( ! apr_file_eof( fd ) )
   {
      if ( result != 0 )
      {
         ap_log_perror( APLOG_MARK, APLOG_ERR, result, cfg->pool,
            "Error while reading file %s at line %d",
            cfg->init_file, line );
         apr_file_close( fd );
         return 0;
      }

      apr_collapse_spaces( buffer, buffer );
      // skip lines to be ignored.
      bool correct = true;
      if ( buffer[0] != 0 && buffer[0] != '#' && buffer[0] != ';' )
      {
         // we have a line.
         char *last;
         char *token;
         token = apr_strtok(buffer,"=",&last);
         if( token == NULL )
         {
            correct = false;
         }

         // Log Errors
         else if ( apr_strnatcasecmp( token, "LogErrors" ) == 0 )
         {
            // set error log mode
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else if (apr_strnatcasecmp( token, "Log" ) == 0 )
               cfg->errorMode = FM_ERROR_MODE_LOG;
            else if (apr_strnatcasecmp( token, "Silent" ) == 0 )
               cfg->errorMode = FM_ERROR_MODE_SILENT;
            else if (apr_strnatcasecmp( token, "Kind" ) == 0 )
               cfg->errorMode = FM_ERROR_MODE_KIND;
            else if (apr_strnatcasecmp( token, "Report" ) == 0 )
               cfg->errorMode = FM_ERROR_MODE_REPORT;
            else
               correct = false;
         }
         // Maximum upload size
         else if ( apr_strnatcasecmp( token, "MaxUpload" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else {
               int size = atoi( token );
               if ( size == 0 && token[0] != '0' )
                  correct = false;
               else
                  cfg->maxUpload = size*1024;
            }
         }
         // Upload mode
         else if ( apr_strnatcasecmp( token, "MaxMemUpload" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else {
               int size = atoi( token );
               if ( size == 0 && token[0] != '0' )
                  correct = false;
               else
                  cfg->maxMemUpload = size*1024;
            }
         }
         // Upload directory
         else if ( apr_strnatcasecmp( token, "UploadDir" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else 
               apr_cpystrn( cfg->uploadDir, token, MAX_UPLOAD_DIR );
         }
         // Falcon path
         else if ( apr_strnatcasecmp( token, "LoadPath" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else
               apr_cpystrn( cfg->loadPath, token, MAX_UPLOAD_DIR );
         }
         else if ( apr_strnatcasecmp( token, "FalconHandler" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else
               apr_cpystrn( cfg->falconHandler, token, MAX_UPLOAD_DIR );
         }
         // Session timeout
         else if ( apr_strnatcasecmp( token, "SessionTimeout" ) == 0 )
         {
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else {
               int to = atoi( token );
               if ( to == 0 && token[0] != '0' )
                  correct = false;
               else
                  cfg->sessionTimeout = to;
            }
         }
         // Session mode
         else if ( apr_strnatcasecmp( token, "SessionMode" ) == 0 )
         {
            // set error log mode
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else if (apr_strnatcasecmp( token, "File" ) == 0 )
               cfg->sessionMode = 1;
            else if (apr_strnatcasecmp( token, "Memory" ) == 0 )
               cfg->sessionMode = 0;
            else
               correct = false;
         }
         else if ( apr_strnatcasecmp( token, "PersistentDataDir" ) == 0 )
         {
            // set error log mode
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else
               apr_cpystrn( cfg->pdataDir, token, MAX_UPLOAD_DIR );
         }
         else if ( apr_strnatcasecmp( token, "CacheModules" ) == 0 )
         {
            // set error log mode
            token = apr_strtok(NULL,"#",&last);
            if ( token == NULL )
               correct = false;
            else if (apr_strnatcasecmp( token, "on" ) == 0 )
               cfg->cacheModules = 1;
            else if (apr_strnatcasecmp( token, "off" ) == 0 )
               cfg->cacheModules = 0;
            else
               correct = false;
         }
         else if ( apr_strnatcasecmp( token, "RunTimeout" ) == 0 )
         {
            // set error log mode
            token = apr_strtok(NULL,"#",&last);
            int to = atoi( token );
            if ( token == NULL )
               correct = false;
            else {
               int timeout = atoi( token );
               if ( timeout == 0 && token[0] != '0' )
                  correct = false;
               else
                  cfg->runTimeout = timeout;
            }
         }
      }

      // Was this line correct?
      if ( ! correct )
      {
         ap_log_perror( APLOG_MARK, APLOG_WARNING, result, cfg->pool,
            "Malformed line in Falcon config file %s at line %d",
            cfg->init_file, line );
      }

      result = apr_file_gets( buffer, 255, fd );
      line++;
   }

   // we're done
   apr_file_close( fd );
   
   /*
   fprintf( stderr, "Falcon WOPI module for Apache2 configuration:\n"
      "-- Loaded: %s\n"
      "-- InitFile: %s\n"
      "-- ErrorMode: %d\n"
      "-- MaxUpload: %d\n"
      "-- MaxMemUpload: %d\n"
      "-- sessionTimeout: %d\n"
      "-- sessionMode: %d\n"
      "-- cacheModules: %s\n"
      "-- runTimeout: %d\n",
      
      cfg->loaded ? "Yes" : "No",
      cfg->init_file,
      cfg->errorMode,
      cfg->maxUpload,
      cfg->maxMemUpload,
      cfg->sessionTimeout,
      cfg->sessionMode,
      cfg->cacheModules ? "Yes" : "No",
      cfg->runTimeout      
      );
    */
   return 1;
}
