/*
   FALCON - The Falcon Programming Language.
   FILE: logging_mod.h

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_logging_mod_H
#define flc_logging_mod_H

#include <falcon/setup.h>
#include <falcon/error_base.h>

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif

#define FALCON_LOGGING_ERROR_OPEN  (FALCON_LOGGING_ERROR_BASE + 0)
#define FALCON_LOGGING_ERROR_DESC  "Error opening the logging service"

#endif

/* end of logging_mod.h */
