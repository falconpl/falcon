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
FALCON_DECLARE_FUNCTION(rest, "time:N");
//FALCON_DECLARE_FUNCTION(epoch, ""); moved in sys
FALCON_DECLARE_FUNCTION(seconds, "");
FALCON_DECLARE_FUNCTION(quit, "value:X");
FALCON_DECLARE_FUNCTION(advance, "collection:X");
FALCON_DECLARE_FUNCTION(int, "item:N|S");
FALCON_DECLARE_FUNCTION(numeric, "item:N|S");
FALCON_DECLARE_FUNCTION(input, "");
FALCON_DECLARE_FUNCTION(passvp, "citem:[C]");
FALCON_DECLARE_FUNCTION(call, "callee:C, params:[A]")

FALCON_DECLARE_FUNCTION(map, "mapper:C, data:X")
FALCON_DECLARE_FUNCTION(filter, "flt:C, data:X")
FALCON_DECLARE_FUNCTION(reduce, "reducer:C, data:X, initial:[X]")

FALCON_DECLARE_FUNCTION(cascade, "callList:A,...")
FALCON_DECLARE_FUNCTION(perform, "&...")
FALCON_DECLARE_FUNCTION(firstOf, "...")
FALCON_DECLARE_FUNCTION(ffirstOf, "&...")
FALCON_DECLARE_FUNCTION(ffor, "initialize:Expression, check:Expression, increment:Expression, code:Syntree")
}
}

#endif	

/* end of stdfunctions.h */
