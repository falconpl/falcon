/*
   FALCON - The Falcon Programming Language.
   FILE: vmtimer.h

   Heart beat timer for processors in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 20 Nov 2012 11:41:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_VMTIMER_H_
#define _FALCON_VMTIMER_H_

#include <falcon/setup.h>
#include <falcon/mt.h>
#include <falcon/pool.h>
#include <falcon/poolable.h>

namespace Falcon {

/**
Heart beat timer for processors in the virtual machine.

@note It must be granted that this class is destroyed after
all its users have canceled their callbacks, or are anyhow
stopped and not receiving them anymore.
*/
class FALCON_DYN_CLASS VMTimer: public Runnable
{
public:
   VMTimer();
   /** Destructor for the timer.
    * This will call stop(), which is a no-op if already called.
    */
   virtual ~VMTimer();

   /** Stops the timer thread.
    *
    * This will wait till the timer thread is fully stopped.
    * This usually happens rapidly, but it might take sometime
    * if the thread is currently engaged in a callback invocation.
    */
   void stop();

   /**
    * Callback class.
    *
    * This class is the type of entities that are called back
    * at timer expiration.
    *
    * The destructor is never called by the VMTimer; if it must be
    * disposed, this must be done by the timer user after the callback
    * has been invoked or safely canceled.
    *
    */
   class Callback {
   public:
      virtual ~Callback() {}
      /**
       * Invokes the required callback.
       *
       * If the callback methods returns true, the token
       * associated with this callback is disposed and becomes
       * invalid. It is advised to do so only if the owner
       * of the callback entity has already abandoned any
       * reference to the token or simply ignores it.
       *
       * @note Callbacks are all called in the timer thread.
       * Performing long computation in this thread will cause other
       * stacked callbacks to lag; instead, perform just minimal
       * operation to let the main thread of the user agent to know
       * about the timeout expiration and return immediately.
       *
       * Notice that nothing prevents the cancel() method of the token
       * to be called while this operator is currently being processed.
       */
      virtual bool operator ()() = 0;
   };

   /**
    * Handler for callback canelation.
    *
    * Other than serving internally to keep track of the
    * callback timeouts, this class is handled back to the
    * timer user as a mean to cancel the timer or declare
    * the data is disposeable again.
    *
    * Tokens are recycled by a pool in this class, so they
    * cannot be deleted by the receiver. Instead, they must
    * have their method dispose() called when the receiver has
    * no more need for them.
    *
    * Note: it is forbidden (and generates an exception) to
    * call the cancel() method from inside the given callback.
    * The timer user must properly dispose the token after the
    * callback has been invoked, or the callback method must return
    * true to ask for automatic disposal of the token.
    */

   class Token: private Poolable
   {
      /** Cancels a pending timeout and/or release the token.
       *
       * The caller must call this method exactly once per received token.
       * If called before the callback is invoked, the timeout is canceled
       * and the callback will not be invoked.
       *
       * @note If the callback is currently being invoked, this method will
       * block until the callback has cleanly exited. This means that the
       * caller must not hold any shared resource that the callback method
       * might be waiting on when cancel() is invoked.
       *
       * After this method returns, the caller should consider the
       * token invalid.
       */
      void cancel();

      /** Gets the associated callback object. */
      Callback* callback() const { return m_cb; }

   private:
      friend class VMTimer;
      Token();
      virtual ~Token();

      Event m_notBusy;
      VMTimer* m_owner;
      Callback *m_cb;

      bool m_canceled;
      Mutex m_mtxCanceled;
   };

   /** Calls a certain callback after a given time.
    * @param ms Count of milliseconds after which to perform the callback.
    * @param cb The callback to be called after the given timeout.
    */
   Token *setTimeout( uint32 ms, Callback* cb );

   /** Calls a certain callback at a given moment in time.
    * @param pointInTime Absolute time when to call the given callback.
    * @param cb The callback to be called after the given timeout.
    */
   Token *setRandezvous( int64 pointInTime, Callback* cb );

   virtual void* run();

private:
   class Private;
   VMTimer::Private* _p;

   friend class Token;
   SysThread* m_thread;

   typedef Pool TokenPool;
   TokenPool m_tokens;

   // mutex for the token pool.
   Mutex m_mtxTokens;

   // Mutex for the internal structures
   Mutex m_mtxStruct;

   InterruptibleEvent m_newWork;

   void checkExpired( int &nextRdv );
};

}

#endif
