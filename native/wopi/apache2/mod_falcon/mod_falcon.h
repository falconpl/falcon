/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_mod.h

   Falcon module for Apache 2
   Global declarations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MOD_H
#define FALCON_MOD_H

#include <httpd.h>
#include <http_config.h>

#define FALCON_FAM_TYPE "application/x-falcon-module"
#define FALCON_FAL_TYPE "application/x-falcon-source"
#define FALCON_FTD_TYPE "application/x-falcon-ftd"
#define FALCON_PROGRAM_HANDLER "falcon-program"
#define FALCON_PROGRAM_FTD     "falcon-ftd"

#ifdef __cplusplus
   #define ext_c extern "C"
#else
   #define ext_c
   #include <aio.h>
#endif

#define DEFAULT_CONFIG_FILE   "/etc/falcon.ini"

ext_c void falcon_register_hook( apr_pool_t *p );
extern module AP_MODULE_DECLARE_DATA falcon_module;

#endif

/* end of mod_falcon.h */

