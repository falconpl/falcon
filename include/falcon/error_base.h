/*
   FALCON - The Falcon Programming Language.
   FILE: error_base.h

   Base error codes for well known modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom feb 18 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

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

#define FALCON_COMPILER_ERROR_BASE        1000
#define FALCON_CONFPARSER_ERROR_BASE      1010
#define FALCON_MXML_ERROR_BASE            1020
#define FALCON_PROCESS_ERROR_BASE         1040
#define FALCON_SOCKET_ERROR_BASE          1060
#define FALCON_ZLIB_ERROR_BASE            1100

#define FALCON_DBI_ERROR_BASE             2000
#define FALCON_THREADING_ERROR_BASE       2050
#define FALCON_SDL_ERROR_BASE             2100

#define FALCON_USER_ERROR_BASE   10000
