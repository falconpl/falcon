/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: systhread_posix.h

   System dependent MT provider - posix provider specific data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Apr 2008 21:32:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent MT provider - posix provider specific data.
*/

#ifndef FLC_SYSTHREAD_POSIX_H
#define FLC_SYSTHREAD_POSIX_H

#include <falcon/setup.h>
#include <pthread.h>

namespace Falcon {
namespace Ext {

class POSIX_THI_DATA: public BaseAlloc
{
public:
   POSIX_THI_DATA();
   ~POSIX_THI_DATA();

   pthread_cond_t m_condSignaled;
   pthread_mutex_t m_mtx;
   volatile bool m_bSignaled;
   volatile bool m_bInterrupted;
   long m_refCount;
};


/** Core of posix oriented waiter implementation.
   The signal/broadcast implementation of WaiterProvider will
   use this class to wake up the correct threads.
*/
class POSIX_WAITABLE: public BaseAlloc
{

public:
   POSIX_WAITABLE( Waitable *w );
   ~POSIX_WAITABLE();

   List m_waiting;
   Waitable *m_waitable;

   /** Try to acquire the resource and eventually wait for it to be free.
      If successful returns true, if unsusccesful the thread is put in
      wait and will be signaled when the resource can be acquired.
   */
   bool waitOnThis( POSIX_THI_DATA *th );
   void cancelWait( POSIX_THI_DATA *th );
};

}
}

#endif

/* end of systhread_posix.h */
