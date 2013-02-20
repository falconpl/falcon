/*
   FALCON - The Falcon Programming Language.
   FILE: semaphore.h

   Falcon core module -- Semaphore shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SEMAPHORE_H
#define FALCON_CORE_SEMAPHORE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedSemaphore: public Shared
{
public:
   SharedSemaphore( ContextManager* mgr, const Class* owner, int32 initCount );
   virtual ~SharedSemaphore();
};


class FALCON_DYN_CLASS ClassSemaphore: public ClassShared
{
public:
   ClassSemaphore();
   virtual ~ClassSemaphore();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:

   /*#
     @method signal Semaphore
     @brief Signals the semaphore
     @optparam count Count of signals to be sent to the semaphore.

      The parameter @b count must be greater or equal to 1.
    */
   FALCON_DECLARE_METHOD( post, "count:[N]" );

   /*#
     @method tryWait Semaphore
     @brief Check if the semaphore is signaled.
     @return true if the semaphore is signaled, false otherwise.

     The check eventually resets the semaphore if it's currently signaled.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
     @method wait Semaphore
     @brief Wait for the semaphore to be  to be open.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the barrier is open during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
};

}


}

#endif	

/* end of semaphore.h */
