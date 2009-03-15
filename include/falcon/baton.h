/*
   FALCON - The Falcon Programming Language.
   FILE: baton.h

   Baton synchronization structure.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 14 Mar 2009 00:03:28 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_BATON_H
#define FLC_BATON_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>

namespace Falcon {

/** Baton concurrency controller class.
   
   Like in a 4x4 relay match, a baton is an object which
   gives the grant to continue operating on a set of objects.
   It is tightly related with the concept of "monitor", but other
   than that, it is possible to force blockade of the runner
   by issuing a block request.

   It is used by VM users and by the GC, which has inspection
   rights that overseed VM execution rights (normally).
*/
class FALCON_DYN_CLASS Baton: public BaseAlloc
{
   void *m_data;

public:
   Baton( bool bBusy = false );
   virtual ~Baton();
   
   /** Acquires the baton.
      
      The caller blocks until it is able to acquire the baton.
      If the baton is blocked, it can be acquired only by the
      blocking thread.
   
      \note Succesful acquisition of the blocking thread causes
            the baton to be automatically unblocked.
   */
   virtual void acquire();
   
   /** Tries to acquire the baton.
      
      If the baton is currently available (unacquired) it is
      acquired, unless blocked. If it is blocked, it can be
      acquired only by the blocker thread.
      
      \note Succesful acquisition of the blocking thread causes
            the baton to be automatically unblocked.
      
      \return true if the baton is acquired.
   */
   bool tryAcquire();
   
   /** Releases the baton.
      This makes the baton available for another acquirer.
   */
   virtual void release();
   
   /** Checks if the baton is blocked, honouring pending block requests.
      
      If the baton is blocked, it is atomically released and the calling
      thread puts itself in wait for re-acquisition.
   */
   void checkBlock();
   
   /** Blocks the baton.
   
      If the call is succesful, this prevents any acquire request coming from
      other threads to be accepted.
      
      Only one thread can block the baton; the call will fail if there is
      an already pending block request issued by another thread. It will succed
      (with no effect) if the caller thread is the one already blocking the
      baton.
      
      \return true on success.
   */
   bool block();
   
   /** Unblocks the baton.
   
      If a succesful blocking thread decides it doesn't want to acquire the
      baton anymore, it can cast an unblock() to allow other acquiring threads
      to progress.
      
      \note A succesful acquisition releases atomically the block request.
      
      \return true if the thread was owning the block onthe baton.
   */
   bool unblock();
   
   /** Debug routine. 
      True if currently busy.
   */
   bool busy();

   /** Function called when the baton is released while there is a pending blocking request.
      The base class version does nothing.
   */
   virtual void onBlockedAcquire();

};

}

#endif

/* end of baton.h */
