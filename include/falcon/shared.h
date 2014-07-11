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
 A base, semaphore-like shared resource as seen by the scheduler and the VM contexts.

 This base resource has a semaphore semantic. It can be posted with more
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
 be exposed, as they could also be used by the extensions transparently with respect
 to the final script.

 As such, is perfectly legal to present a Shared without a class handler (handler = 0),
 in case the shared is completely controlled by reachable code and needs not to be
 handled by the garbage collector.

 If the resource must be assigned to the garbage collector,
 the ClassShared* handler provided in the Engine::stdHandlers() can be used.
 */
class FALCON_DYN_CLASS Shared
{
public:
   Shared( ContextManager* mgr, const Class* handler=0, bool acquireable = false, int32 signals = 0 );

   /** Returns true if this resource supports acquire semantic.
    */
   bool hasAcquireSemantic() const { return m_acquireable; }
   void dropWaiting( VMContext* ctx );

   virtual Shared* clone() const;

   /**
    Consumes one or more signals atomically.

    Might consume less signals than required if there are less
    signals available in the shared resource.

    \param count Number of signals that the caller wants to consume.
    \return The number of signals really consumed.

    */
   virtual int32 consumeSignal( VMContext* target, int32 count = 1 );
   /** Post one or more signals signal to this resource.

   \param count Number of signals posted to this resource.

   The first signal \b only sends a request to the context manager to
   publish the information.

   This is consistent if the manager
   can atomically decrease the signals as the waiters are
   dequeued. Not all the shared resources can grant this
   semantic; in that case, the signal method must be
   reimplemented to notify the manager also in other
   occasions.
    */
   virtual void signal( int count = 1 );

   /**
    Posts a number of signals equal to the current number of waiters.

    Normally, this method is to be avoided, but is provided for
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
   const Class* handler()  const { return m_cls; }


   virtual uint32 currentMark() const { return m_mark; }
   virtual void gcMark( uint32 n ) { m_mark = n; }

   /**
    * Called back when a waiter context wants to be notified about changes.
    *
    * The base class version does nothing.
    */
   virtual void onWaiterAdded(VMContext* ctx);

   /**
    * Invoked right before the target context engages a wait on this resource.
    *
    * The base class version does nothing. This is usually not called
    * for a try-wait (to = 0).
    */
   virtual void onWaiterWaiting(VMContext* ctx, int64 to);

   /**
    * Determines whether this resource is context-specific.
    *
    * Context specific resources are consumed by context.
    * This means that the context manager cannot presume that the
    * resource cannot have more signals for other contexts when it
    * finds that it has not any signal ready anymore for a specific
    * context.
    *
    * With context specific resources, the manager must try a signaled
    * resource on all the sleeping contexts.
    *
    */
   bool isContetxSpecific() const { return m_bContextSpec; }

protected:

   uint32 m_mark;
   bool m_bContextSpec;

   virtual ~Shared();

   /** Returns the count of the signals in this moment.
    *
    * Used by subclasses that don't implement a semaphore semantic.
    */
   int32 signalCount() const;

   virtual int32 lockedConsumeSignal( VMContext* target, int32 count = 1 );
   virtual void lockedSignal( int32 count = 1 );
   virtual int32 lockedSignalCount() const;

   void lockSignals() const;
   void unlockSignals() const;


   /**
    * Called back by the context manager after a signaled shared resource is processed.
    *
    * This is called back during the signal-lock loop, so the "locked" version of
    * methods should be used.
    */
   virtual void onWakeupComplete();

   ContextManager* notifyTo() const { return m_notifyTo; }

private:
   ContextManager* m_notifyTo;
   class Private;
   Private* _p;

   bool m_acquireable;
   const Class* m_cls;

   friend class ContextManager;
   friend class VMContext;

   bool addWaiter( VMContext* ctx );


   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Shared);
};

}

#endif

/* end of shared.h */

