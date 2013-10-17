/*
   FALCON - The Falcon Programming Language.
   FILE: errors.h

   Error for WOPI exceptions.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Mar 2010 17:12:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef WOPI_ERROR_H
#define WOPI_ERROR_H

#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_ERROR_WOPI_BASE
   #define FALCON_ERROR_WOPI_BASE            2400
#endif

#define FALCON_ERROR_WOPI_SESS_IO            (FALCON_ERROR_WOPI_BASE + 0 )
#define FALCON_ERROR_WOPI_SESS_EXPIRED       (FALCON_ERROR_WOPI_BASE + 1 )
#define FALCON_ERROR_WOPI_APPDATA_SER        (FALCON_ERROR_WOPI_BASE + 2 )
#define FALCON_ERROR_WOPI_APPDATA_DESER      (FALCON_ERROR_WOPI_BASE + 3 )
#define FALCON_ERROR_WOPI_SESS_INVALID_ID    (FALCON_ERROR_WOPI_BASE + 4 )
#define FALCON_ERROR_WOPI_PERSIST_NOT_FOUND  (FALCON_ERROR_WOPI_BASE + 5 )
#define FALCON_ERROR_WOPI_FIELD_NOT_FOUND    (FALCON_ERROR_WOPI_BASE + 6 )

#endif

/* end of errors.h */
