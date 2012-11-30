/*
   FALCON - The Falcon Programming Language.
   FILE: shared.h

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SHARED_H_
#define _FALCON_SHARED_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/refcounter.h>

namespace Falcon {
class VMContext;
class ContextManager;
class Class;

/**
 An abstract shared resource as seen by the scheduler and the VM contexts.

 Shared resources have a semaphore semantic. They can be posted with more
 signals, that are then consumed by the waiters. When there are no more signals
 to be consumed, waiters are forced to wait for some signal to become available.

 Falcon shared resources have also an acquireable semantic: if a resource is
 acquireable, the signaled VMContext acquires it, and automatically post a signal
 when:
 - It is stopped due to an execption or to normal termination.
 - It enters a wait.
 - It is descheduled due to explicit request (sleep).

 Also, a context having acquired a resource won't be descheduled on time-slice expiration,
 becoming a "super-coroutine" context while it holds the resource.

 Acquireable shared resources could theoretically be acquired by more than one context,
 provided they have enough signals available. This is usually not the case, but some exotic
 resource might be acquireable by more contexts without any particular consequence for
 the parallel model.

 A single-signalable shared resource with acquire semantic is a coroutine-wise mutex.

 The signal and consumeSignal methods are virtual; some subclasses might operate with
 a quasi-semaphore semantic. It's the case of the Barrier, that can be in open or closed
 state, with a wait operation that doesn't consume any signal,
 and it's typically used to signal an asynchronous program termination request.

 Shared subclasses should present a Class handler that is then used by the Falcon engine
 to expose the resource to the scripts. However, concrete shared resources need not to
 be exposed, as they could also be used by the
 */
class FALCON_DYN_CLASS Shared
{
public:
   Shared( Class* cls=0, bool acquireable = false, int32 signals = 0 );

   /** Returns true if this resource supports acquire semantic.
    */
   bool hasAcquireSemantic() const { return m_acquireable; }
   void addWaiter( VMContext* ctx );
   void dropWaiting( VMContext* ctx );

   virtual Shared* clone() const;

   /**
    Consumes one or more signals atomically.

    Might consume less signals than required if there are less
    signals available in the shared resource.

    \param count Number of signals that the caller wants to consume.
    \return The number of signals really consumed.

    */
   virtual int32 consumeSignal( int32 count = 1 );
   /** Post one or more signals signal to this resource.

   \param count Number of signals posted to this resource.
    */
   virtual void signal( int count = 1 );

   /**
    Posts a number of signals equal to the current number of waiters.

    Normally, this request is to be avoided, but is provided for
    special purposes.

    A program relying on this semantic must take measures so that:
    # Waiters cannot wake up during the wake-up request;
    # No new subscribers can possibly enter in wait before all the waiters are
      signaled, or, if they do, they are signaled through other means;
    # No other signaler can post any signal while waiters are being scheduled.

    The first condition is safely met if all the waiters have endless wait.
    The second condition is safely met if the broadcast can only happen after all
    the possible waiters have safely entered the wait.
    The third condition is safely met if there just one broadcaster.

    Actually, this method can also be used if the above conditions are not met,
    but only for special purpose algorithms and only provided you Know What You're
    Doing(TM).

    \note This broadcast is \b NOT the same thing as a POSIX condition broadcast.
    Doesn't work the same way, doesn't offer the same guarantees and cannot be used
    in the same patterns.

    \note Once you Know What You're Doing(TM), be sure to check any other means to
    reach the same result twice or thrice.

    */
   virtual void broadcast();

   /** Returns the VM class handler associated with this resource instance.
    \return The handler class for this shared object, or 0 if the object cannot
    be handled by the engine.
    */
   Class* handler()  const { return m_cls; }

   uint32 gcMark() const { return m_mark; }
   void gcMark( uint32 n ) { m_mark = n; }

private:
   class Private;
   Private* _p;

   bool m_acquireable;
   Class* m_cls;

   friend class ContextManager;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Shared);

   virtual ~Shared();

   uint32 m_mark;
};

}

#endif

/* end of shared.h */

