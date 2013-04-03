/*
   FALCON - The Falcon Programming Language.
   FILE: foreach.h

   Falcon core module -- foreach function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Apr 2013 00:46:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_FOREACH_H
#define FALCON_CORE_FOREACH_H

#include <falcon/function.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function foreach
   @brief iterates over a generator or iterable sequence.
   @param sequence iterable sequence
   @param code The executable code to which each item in @b sequence is passed.
   @return the last evaluation result from @b code
*/

/*#
   @method foreach BOM
   @brief iterates over a this generator or iterable sequence.
   @param code The executable code to which each item in @b sequence is passed.
   @return the last evaluation result from @b code
   @see foreach
*/

class FALCON_DYN_CLASS Foreach: public Function
{
public:
   Foreach();
   virtual ~Foreach();
   virtual void invoke( VMContext* vm, int32 nParams );
};

}
}

#endif	/* FALCON_CORE_FOREACH_H */

/* end of foreach.h */
