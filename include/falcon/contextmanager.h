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
#include <falcon/types.h>
#include <falcon/mt.h>
#include <falcon/syncqueue.h>
#include <falcon/shared.h>

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
    Invoked by a group when it is being terminated because of errors.

    This just inform the manager that the context is scheduled for future termination.
    Normally, the context will arrive to the manager at a later time, with a
    deschedule message and terminate event.

    However, if the context is currently sleeping (possibly forever), it must be
    removed by the context manager. This message asks the context manager to
    remove the context on its initiative, if it is still sleeping.
    */
   void onGroupTerminated( VMContext* ctx );

   /**
   Called back when a shared resource with contexts in wait is signaled.
    */
   void onSharedSignaled( Shared* waitable );

   /**
    * Used by the collector to wake up sleeping contexts.
    */
   void wakeUp(VMContext* ctx);

   /** Main loop of the context manager agent. */
   virtual void* run();

   /** Called when a context has been registered with the collector and is ready to go */
   void onContextReady(VMContext* ctx);

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


   typedef SyncQueue<VMContext*> ReadyContextQueue;
   ReadyContextQueue& readyContexts() { return m_readyContexts; }

private:

   SysThread* m_thread;
   int64 m_next_schedule;
   int64 m_now;

   Mutex m_mtxStopped;
   bool m_bStopped;

   class Private;
   ContextManager::Private* _p;

   bool manageSleepingContexts();

   void manageContextKilledInGroup( VMContext* ctx );
   void manageTerminatedContext( VMContext* ctx );
   void manageDesceduledContext( VMContext* ctx );
   void manageReadyContext( VMContext* ctx );
   void manageSignal( Shared* ctx );
   void manageAwakenContext( VMContext* ctx );

   bool removeSleepingContext( VMContext* ctx );

   //==============================================
   // Context ready to be scheduled.
   //
   ReadyContextQueue m_readyContexts;
};

}

#endif

/* end of contextManager.h */

