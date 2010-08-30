/*
   FALCON - The Falcon Programming Language.
   FILE: wopi_ext.h

   Falcon Web Oriented Programming Interface.

   Main module generator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 16:19:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_EXT_H
#define FALCON_WOPI_EXT_H

#include <falcon/module.h>

namespace Falcon{
namespace WOPI {

Falcon::Module *wopi_module_init( ObjectFactory rqf, ObjectFactory rpf,
      ext_func_t rq_init_func=0, ext_func_t rp_init_func=0  );

}
}

#endif

/* end of wopi_ext.h */
