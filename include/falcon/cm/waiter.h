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
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon {
class Shared;

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
class ClassWaiter: public Class
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
};

}
}


#endif

/* end of waiter.h */
