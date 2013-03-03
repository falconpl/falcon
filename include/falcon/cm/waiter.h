/*
   FALCON - The Falcon Programming Language.
   FILE: waiter.h

   Falcon core module -- Object helping to wait on repeated shared
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Nov 2012 13:52:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WAITER_H_
#define _FALCON_WAITER_H_

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @class Waiter
   @brief Structure supporting repeated waits on multiple objects.
   @param ... Zero or more shared object to wait on at a later moment.
   @ingroup parallel

   This structure supports waits on an arbitrary number of shared resources.
   This is similar to @a Parallel.wait, but the waiter also rotates efficiently
   all the waited objects, giving more priority to a different shared entity
   at each @a Waiter.wait invocation.

   Shared entities can be added immediately at creation or at a later moment.

   @code
      mutex = Mutex()
      sem = Semaphore()
      bar = Barrier()
      queue = SyncQueue()

      ...
      wt = Waiter( mutex, sem, bar )
      wt.add( queue )
      ...
      loop
         signaled = wt.wait()
         switch wt
            case mutex
               ...
            case sem
               ...
            case bar
               ...
            case queue
               ...
         end
      end
   @endcode

   @note The presence of a shared resource in the waiting set can be
   tested using the @b in operator.

   @prop len Number of shared object currently held.
*/
class ClassWaiter: public ClassUser
{
public:
   ClassWaiter();
   virtual ~ClassWaiter();

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   void store( VMContext*, DataWriter*, void* ) const;
   void restore( VMContext*, DataReader*) const;
   void flatten( VMContext*, ItemArray&, void* ) const;
   void unflatten( VMContext*, ItemArray&, void* ) const;

   //=============================================================
   //
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   virtual void op_in( VMContext* ctx, void* instance ) const;

   void internal_wait( VMContext* ctx, int64 to );

private:

   void returnOrInvoke( VMContext* ctx, Shared* sh );

   FALCON_DECLARE_INTERNAL_PSTEP(AfterWait);
   FALCON_DECLARE_INTERNAL_PSTEP(AfterCall);

   FALCON_DECLARE_PROPERTY( len );

   /*#
       @method wait Waiter
       @brief Waits on all the shared resources added up to date.
       @optparam timeout Number of milliseconds to wait for an event.
       @return The signaled resource or nil at timeout.

       If @b timeout is -1, or not given, waits indefinitely. If it's zero,
       it just checks for any signaled resource and then exits. If it's
       greater than zero, waits for the required amount of time.

       If the timeout expires before the resource is signaled, this method
       returns nil.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );

   /*#
       @method tryWait Waiter
       @brief Checks if any of the shared resources added up to date is currently signaled.
       @return The signaled resource or nil if none is signaled.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
       @method add Waiter
       @brief Adds a resource to the wait set.
       @param shared The resource to be added.
       @return The @b self object to allow chaining more add methods.

    */
   FALCON_DECLARE_METHOD( add, "shared:Shared" );
   /*#
       @method remove Waiter
       @brief Removes a resource from the wait set.
       @param shared The resource to be removed.

       If the resource is not present in the waiting set,
       the method silently fails.
    */
   FALCON_DECLARE_METHOD( remove, "shared:Shared" );
};

}
}


#endif

/* end of waiter.h */
