/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

   Falcon VM interface to logging module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/



#ifndef FLC_LOGGING_EXT_H
#define FLC_LOGGING_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

#include <falcon/error_base.h>

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif
/*
#define FALCP_ERR_INVFORMAT  (FALCON_CONFPARSER_ERROR_BASE + 0)
#define FALCP_ERR_STORE      (FALCON_CONFPARSER_ERROR_BASE + 1)
*/
namespace Falcon {
namespace Ext {

// ==============================================
// Class LogArea
// ==============================================
FALCON_FUNC  LogArea_init( ::Falcon::VMachine *vm );

}
}

#endif

/* end of logging_ext.h */
