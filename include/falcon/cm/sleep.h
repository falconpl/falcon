/*
   FALCON - The Falcon Programming Language.
   FILE: sleep.h

   Falcon core module -- VMContext execution suspension
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 22 Jan 2013 11:24:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SLEEP_H
#define FALCON_CORE_SLEEP_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function sleep
   @brief Put the current coroutine at sleep for some time.
   @param time Time, in seconds and fractions, that the coroutine wishes to sleep.
   @return an item posted by the embedding application.

   This function declares that the current coroutines is not willing to proceed at
   least for the given time. The VM will swap out the coroutine until the time has
   elapsed, and will make it eligible to run again after the given time lapse.

   The parameter may be a floating point number if a pause shorter than a second is
   required.
*/

class FALCON_DYN_CLASS Sleep: public Function
{
public:   
   Sleep();
   virtual ~Sleep();
   virtual void invoke( VMContext* ctx, int32 nParams );
};

}
}

#endif	

/* end of sleep.h */
