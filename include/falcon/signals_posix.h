/*
   FALCON - The Falcon Programming Language.
   FILE: signals_posix.h

   POSIX-specific signal handling
   -------------------------------------------------------------------
   Author: Jan Dvorak
   Begin: Fri, 12 Feb 2010 11:23:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SIGNALS_POSIX_H
#define FALCON_SIGNALS_POSIX_H

/* \file
   POSIX signals handling.
*/

#include <falcon/vm.h>
#include <falcon/vmmsg.h>
#include <falcon/mt.h>
#include <signal.h>

namespace Falcon {

/*
 * POSIX signal to VM Message translator.
 */
class FALCON_DYN_CLASS SignalReceiver: public Runnable, public BaseAlloc
{
protected:
   /*
    * VM to send messages based on signals to.
    */
   VMachine *m_targetVM;

   /*
    * Thread waiting for signals to be delivered as long
    * as the shallRun is true.
    */
   SysThread *m_thread;
   bool m_shallRun;

   /*
    * Wait for signals in ->sigset while ->shallRun is true.
    */
   void *run();

   /*
    * Function called from our signal handler to deliver
    * signal information to the helper thread.
    */
   void deliver(int signum, siginfo_t *siginfo);

   /*
    * Wake up the helper thread to check ->shallRun.
    */
   void wakeup(void);

   /*
    * For access to ->deliver().
    */
   friend void signal_handler(int signum, siginfo_t *siginfo, void *ctx);

public:
   /*
    * Initializes signal receiver with a target VM.
    */
   SignalReceiver(VMachine *targetVM);

   /*
    * Waits for the helper thread to stop.
    */
   ~SignalReceiver();

   /*
    * Starts the helper thread.
    */
   void start();

   /*
    * Asks thread to exit and waits for that.
    */
   void stop();

   /*
    * Trap given signal.
    */
   bool trap(int signum);

   /*
    * Stop traping given signal.
    */
   bool reset(int signum);
};

/*
 * Global signal receiver instance.
 */
extern SignalReceiver *signalReceiver;

}

#endif

// vim: et ts=3 sw=3 :
/* end of signals_posix.h */
