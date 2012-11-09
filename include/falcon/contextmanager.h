/*
   FALCON - The Falcon Programming Language.
   FILE: contextmanager.h

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CONTEXTMANAGER_H_
#define _FALCON_CONTEXTMANAGER_H_

#include <falcon/setup.h>

namespace Falcon {
class Shared;
class VMContext;

/**
 VM Scheduler managing waits and sleeps of contexts.
 */
class FALCON_DYN_CLASS ContextManager: public Runnable
{
public:
   ContextManager();
   virtual ~ContextManager();

   /** Puts a context in wait for the required resources or at sleep.

   This method is called back when a context is desceduled by a processor.

    \param ctx The context put at sleep.
   */
   void onContextDescheduled( VMContext* ctx );

   /**
    Called back when a (possibly) descheduled context is being asynchronously terminated.
    \param ctx the context that is being terminated.

    This might happen when the context group where this context belongs is killed.

    The method does something only in the case the context is actually currently held
    by the scheduler, in wait for being posted to the sleeping contexts set or currently
    in there. In that case the method returns true.
    */
   void onContextTerminated( VMContext* ctx );

   /**
   Called back when a shared resource with contexts in wait is signaled.
    */
   void onSharedSignaled( Shared* waitable );

   /** Main loop of the context manager agent. */
   virtual void* run();

   /** Launches the manager thread. */
   bool start();

   /** Terminates the manager thread.
    This should be called right before manager destruction by the VM.
    */
   bool stop();

   /** Utility method to transform a relative wait time into an absolute randez-vous.
    \param to The timeout from now in milliseconds.
    \return The absolute randez-vous in time.
    * */
   static int64 msToAbs( int32 to );

   /** Returns the next ready context, putting the caller in wait if there aren't more ready contexts.
       \return a valid context or 0 if the VM is being terminated, or if the to is expired.
       \param terminateHandler must be set to a 0 integer in input; will be 1 if
          the caller is terminated.

       When stop() is invoked, all the waiting processors are waken up and terminated.
       The invoker can be terminated (or just waken up)
       by asynchronously calling terminateWaiterForReadyContext.

       The invoker can be
    */
   VMContext* getNextReadyContext( int* terminateHandler );

   /** Terminates a single waiter, waking it up and setting the terminate handler to true.
    The owner of the termination handler doesn't really need to be currently engaged
    in wait. In that case, nothing is done (just, you have a spurious wake up of other
    waiters that will be sent back to sleep).
    */
   void terminateWaiterForReadyContext( int* terminateHandler );

private:

   SysThread* m_thread;
   int64 m_next_schedule;
   int64 m_now;

   Mutex m_mtxStopped;
   bool m_bStopped;

   class Private;
   Private* _p;

   bool manageSleepingContexts();
};

}

#endif

/* end of contextManager.h */

