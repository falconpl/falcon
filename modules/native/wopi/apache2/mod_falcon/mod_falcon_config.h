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

#ifndef MOD_FALCON_CONFIG
#define MOD_FALCON_CONFIG

#include "mod_falcon.h"

#define FM_ERROR_MODE_SILENT 0
#define FM_ERROR_MODE_LOG 1
#define FM_ERROR_MODE_REPORT 2
#define FM_ERROR_MODE_KIND 3

#define MAX_LOAD_PATH          1024
#define FM_DEFAULT_SESSION_TO   600
#define FM_DEFAULT_SESSION_MODE 1

#define DEFAULT_HANDLER_NAME    "handler.fal"

#ifdef WIN32
   #define FM_DEFAULT_UPDIR "C:/TEMP"
#else
   #define FM_DEFAULT_UPDIR "/tmp"
#endif

typedef struct
{
   apr_pool_t *pool;
   int loaded;
   char init_file[MAX_LOAD_PATH];
   char falconHandler[MAX_LOAD_PATH];

   void* templateWopi;
   void* errHand;

} falcon_mod_config;

typedef struct
{
   char init_file[MAX_LOAD_PATH];
   char falconHandler[MAX_LOAD_PATH];

   void* templateWopi;
   void* errHand;
} falcon_dir_config;


// Our global configuration settings.
extern falcon_mod_config *the_falcon_config;

ext_c void *falcon_mod_create_config(apr_pool_t *p, server_rec *s);
ext_c void *falcon_mod_create_dir_config(apr_pool_t *p, char *s);
ext_c void *falcon_mod_merge_dir_config(apr_pool_t *p, void* BASE, void* ADD );

ext_c const char *falcon_mod_set_config(cmd_parms *parms, void *mconfig, const char *arg);
ext_c const char *falcon_mod_set_handler(cmd_parms *parms, void *mconfig, const char *arg);

int falcon_mod_load_config( falcon_mod_config *cfg );

#endif
