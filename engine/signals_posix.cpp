/*
   FALCON - The Falcon Programming Language.
   FILE: signals_posix.cpp

   POSIX-specific signal handling
   -------------------------------------------------------------------
   Author: Jan Dvorak
   Begin: Fri, 12 Feb 2010 11:23:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/signals_posix.h>
#include <unistd.h>

namespace Falcon {

SignalReceiver *signalReceiver = 0;

void signal_handler(int signum, siginfo_t *siginfo, void *ctx)
{
   signalReceiver->deliver(signum, siginfo);
}

SignalReceiver::SignalReceiver(VMachine *targetVM):
   m_thread(0),
   m_shallRun(false)
{
   m_targetVM = targetVM;
}

SignalReceiver::~SignalReceiver()
{
   stop();
}

void SignalReceiver::start()
{
   if (0 != m_thread)
      return;

   m_thread = new SysThread(this);
   m_shallRun = true;
   m_thread->start();
}

void SignalReceiver::stop()
{
   if (0 == m_thread)
      return;

   void *dummy;
   m_shallRun = false;
   wakeup();
   m_thread->join(dummy);
   m_thread = 0;
}

void *SignalReceiver::run()
{
   sigset_t sigset;

   /*
    * Allow all signals in this thread.
    */
   sigemptyset(&sigset);
   sigaddset(&sigset, SIGCONT);
   pthread_sigmask(SIG_SETMASK, &sigset, NULL);

   /* Wait... and wait... and wait. */
   while (m_shallRun)
      sigwaitinfo(&sigset, NULL);

   return 0;
}

bool SignalReceiver::trap(int signum)
{
   struct sigaction sigaction;

   sigaction.sa_sigaction = signal_handler;
   sigfillset(&sigaction.sa_mask);
   sigaction.sa_flags = SA_SIGINFO;

   return 0 == ::sigaction(signum, &sigaction, 0);
}

bool SignalReceiver::reset(int signum)
{
   return 0 == ::signal(signum, SIG_DFL);
}

void SignalReceiver::deliver(int signum, siginfo_t *siginfo)
{
   VMMessage *m = new VMMessage("os.signal");
   m->addParam(SafeItem((int32)signum));
   m_targetVM->postMessage(m);
}

void SignalReceiver::wakeup(void)
{
   kill(getpid(), SIGCONT);
}

void BlockSignals()
{
   sigset_t sigset;
   sigfillset(&sigset);
   pthread_sigmask(SIG_SETMASK, &sigset, NULL);
}

void UnblockSignals()
{
   sigset_t sigset;
   sigemptyset(&sigset);
   pthread_sigmask(SIG_SETMASK, &sigset, NULL);
}

}

// vim: et ts=3 sw=3 :
/* end of signals_posix.cpp */
