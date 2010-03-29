/*
   FALCON - The Falcon Programming Language.
   FILE: error_base.h

   Base error codes for well known modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 May 2008 13:55:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base error codes for well known modules.

   This file contains the defines used as "base error codes"
   by the well known modules (Falcon feathers and other featured modules)
   authorized to use this resource.

   Each macro defines an base value for error codes that is then used by the
   module to declare unique vaules.

   Codes below 1000 are reserved for the engine, and below 2000 are reserved for
   feathers. Codes above 10000 (FALCON_USER_ERROR_BASE) are granted to
   be available for user applications.
*/

#ifndef flc_error_base_H
#define flc_error_base_H

#define FALCON_RTL_ERROR_BASE             1000
#define FALCON_COMPILER_ERROR_BASE        1100
#define FALCON_CONFPARSER_ERROR_BASE      1110
#define FALCON_MXML_ERROR_BASE            1120
#define FALCON_PROCESS_ERROR_BASE         1140
#define FALCON_REGEX_ERROR_BASE           1160
#define FALCON_SOCKET_ERROR_BASE          1170
#define FALCON_ZLIB_ERROR_BASE            1190
#define FALCON_LOGGING_ERROR_BASE         1200
#define FALCON_JSON_ERROR_BASE            1210

#define FALCON_DBI_ERROR_BASE             2000
#define FALCON_THREADING_ERROR_BASE       2050
#define FALCON_SDL_ERROR_BASE             2100
#define FALCON_PDF_ERROR_BASE             2200
#define FALCON_ERROR_DYNLIB_BASE          2250
#define FALCON_ERROR_DBUS_BASE            2300
#define FALCON_ERROR_GD_BASE              2330
#define FALCON_ERROR_CURL_BASE            2350
#define FALCON_ERROR_WOPI_BASE            2400

#define FALCON_USER_ERROR_BASE   10000

#endif

/* end of error_base.h */
