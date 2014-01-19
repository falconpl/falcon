/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon.c
   $Id$

   Falcon module for Apache 2
   Main plugin file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "mod_falcon.h"
#include "mod_falcon_config.h"

/*=========================================================
  Module data.
  Configuration directive for Falcon module
*/
static const command_rec mod_falcon_cmds[] =
{
   AP_INIT_TAKE1(
      "FalconConfig",
      (const char *(*)())falcon_mod_set_config,
      NULL,
      RSRC_CONF,
      "config_file (string) -- Location of the falcon.ini file."
   ),
   
   AP_INIT_TAKE1(
      "FalconHandler",
         (const char *(*)())falcon_mod_set_handler,
         NULL,
         ACCESS_CONF | RSRC_CONF,
         "handler script (string) -- Program invoked when falcon-program handler is excited."
         ),


     AP_INIT_TAKE1(NULL, NULL, NULL, 0, NULL )
};

module AP_MODULE_DECLARE_DATA falcon_module = {
   STANDARD20_MODULE_STUFF,
   falcon_mod_create_dir_config,    /* create per-directory config structure */
   falcon_mod_merge_dir_config,     /* merge per-directory config structures */
   falcon_mod_create_config,        /* create per-server config structure */
   NULL,                  /* merge per-server config structures */
   mod_falcon_cmds,       /* command apr_table_t */
   falcon_register_hook   /* register hooks */
};

/* end of mod_falcon.c */
