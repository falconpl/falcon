/*
   FALCON - The Falcon Programming Language.
   FILE: request_ext.cpp

   Web Oriented Programming Interface

   Request class script interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Feb 2010 13:27:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_REQUEST_EXT_H
#define _FALCON_WOPI_REQUEST_EXT_H

#include <falcon/engine.h>

namespace Falcon {
namespace WOPI {

void InitRequestClass( Module* m, ObjectFactory cff, ext_func_t init_func = 0 );
}
}

#endif

/* end of request_ext.h */
