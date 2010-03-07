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
#include <falcon/lineardict.h>
#include <unistd.h>
#include <string.h>

#if !defined(__linux__) || !defined(__sun)
# define __BSD__
#endif	

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
   {
   #ifndef __APPLE__
      sigwaitinfo(&sigset, NULL);
   #else
      sigwait(&sigset, NULL);   
   #endif
   }
   
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
   LinearDict *ld = new LinearDict();
   CoreDict *cd = new CoreDict(ld);

   ld->put(new CoreString("signo"), (int32)signum);
   ld->put(new CoreString("errno"), (int32)siginfo->si_errno);
   ld->put(new CoreString("code"), (int32)siginfo->si_code);

#ifndef __APPLE__
   if (SIGCHLD == signum || (signum >= SIGRTMIN && signum <= SIGRTMAX)) {
      ld->put(new CoreString("pid"), (int32)siginfo->si_pid);
      ld->put(new CoreString("uid"), (int32)siginfo->si_uid);
#ifndef __BSD__
      ld->put(new CoreString("utime"), (int64)siginfo->si_utime);
      ld->put(new CoreString("stime"), (int64)siginfo->si_stime);
#endif
   }
   if (signum >= SIGRTMIN && signum <= SIGRTMAX) {
#ifdef _POSIX_SOURCE
      ld->put(new CoreString("overrun"), (int32)siginfo->si_overrun);
      ld->put(new CoreString("timerid"), (int32)siginfo->si_timerid);
      ld->put(new CoreString("int"), (int32)siginfo->si_int);
#endif
   }
   if (SIGCHLD == signum) {
      ld->put(new CoreString("status"), (int32)siginfo->si_status);
   }
#ifndef __BSD__
   if (SIGPOLL == signum) {
      ld->put(new CoreString("band"), (int32)siginfo->si_band);
      ld->put(new CoreString("fd"), (int32)siginfo->si_fd);
   }
#endif
#else /* MACOSX */
      ld->put(new CoreString("pid"), (int32)siginfo->si_pid);
      ld->put(new CoreString("uid"), (int32)siginfo->si_uid);
      ld->put(new CoreString("status"), (int32)siginfo->si_status);
      ld->put(new CoreString("band"), (int32)siginfo->si_band);
#endif

   cd->bless(true);

   m->addParam(SafeItem(cd));
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
