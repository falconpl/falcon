/*
   FALCON - The Falcon Programming Language.
   FILE: scheduler.h

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SCHEDULER_H_
#define _FALCON_SCHEDULER_H_

#include <falcon/setup.h>

namespace Falcon {
class Shared;
class VMContext;

/**
 VM Scheduler managing waits and sleeps of contexts.
 */
class FALCON_DYN_CLASS Scheduler
{
public:
   Scheduler();
   virtual ~Scheduler();

   /** Adds a context immediately scheduling it for start. */
   void addContext( VMContext *ctx );

   /** Puts a context at sleep for a given time.

    The sleeping context is not receiving message updates.
    \param ctx The context put at sleep.
    \param timeout Wait before being waken in microseconds.
   */
   void putAtSleep( VMContext* ctx, uint32 timeout );

   /** Puts a context in wait for the required resources.

    The required resources are declared by the context prior
    being put in wait.

    The context is awaken when one or more resource get signaled.

    \param ctx The context put at sleep.
    \param timeout Wait before being waken in microseconds.
   */
   void putInWait( VMContext* ctx, time_t wakeupTime );

   /** Puts a context in wait for the required resources indefinitely.

    The context is not put in the sleeping context map, so it will
    never got awaken until some of the resources the context is waiting
    for get signaled.

    \param ctx The context put in wait.
   */
   void putInWait( VMContext* ctx );

   /**
    Terminates a context.

    The context is removed from the scheduler, but still not destroyed.
     \param ctx The context put in wait.
    */
   void terminateContext( VMContext *ctx );

   /**
    Signals a shared resource waking up the waiting contexts.

    The resource is automatically and atomically released
    */
   void signalResource( Shared* waitable );

   void broadcastResource( Shared* waitable );

   /**
    Sets a processor count.
    */
   void setProcessors( int32 count );

   /**
    * Stops the scheduler (and waits for all the processors to be terminated).
    */
   void stop();

   /**
    Asynchronous manager for the processor.

    This separate thread linearizes and then arbitrates the acquisition
    of resources by contexts, and their wakeups.

    */
   class FALCON_DYN_CLASS Manager: public Runnable
   {
      Manager( Scheduler* owner );
      virtual ~Manager();

      virtual void* run();
      void join() { void* dummy = 0; m_thread->join(dummy); }

      /**
       Message exchanged between the manager and the scheduler base class.
       */
      class FALCON_DYN_CLASS Msg {
      public:
         typedef enum {
            e_terminate_context,
            e_sleep_context,
            e_wait_context,
            e_signal_resource,
            e_aquirable_resource,
            e_stop
         }
         t_type;

         /** Type of message */
         t_type m_type;
         /** data traveling with the message */
         union {
            VMContext* ctx;
            Shared* res;
         }
         m_data;

         /** Absolute wake-up time for contexts put at sleep or in wait. */
         int64 m_absTime;

         Msg( t_type type ):
            m_type(type) {}

         Msg( t_type type, VMContext* ctx, int64 to = 0 ):
            m_type(type), m_absTime(to) { m_data.ctx = ctx; }

         Msg( t_type type, Shared* res ):
            m_type(type) { m_data.res = res; }
      };

   private:

      Scheduler* m_owner;
      SysThread* m_thread;
      int64 m_next_schedule;
      int64 m_now;

      bool manageSleepingContexts();

      void onTerminateContext( VMContext* ctx );
      void onSleepContext( VMContext* ctx, int64 absTime );
      void onWaitContext( VMContext* ctx, int64 absTime );
      void onSignal( Shared* res );
      void onAcquirable( Shared* res );
   };

private:

   class Private;
   Private* _p;
};

}

#endif

/* end of scheduler.h */

