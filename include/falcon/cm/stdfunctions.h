/*
   FALCON - The Falcon Programming Language.
   FILE: stdfunctions.h

   Falcon core module -- Standard functions found in the core module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 22 Jan 2013 11:24:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_STDFUNCTIONS_H
#define FALCON_CORE_STDFUNCTIONS_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

FALCON_DECLARE_FUNCTION(sleep, "time:N");
FALCON_DECLARE_FUNCTION(epoch, "");
FALCON_DECLARE_FUNCTION(seconds, "");
FALCON_DECLARE_FUNCTION(advance, "collection:X");

}
}

#endif	

/* end of sleep.h */
