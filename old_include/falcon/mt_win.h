/*
   FALCON - The Falcon Programming Language.
   FILE: mt_win.h

   Multithreaded extensions - MS-Windows specific header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 13:08:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MT_WIN_H
#define FALCON_MT_WIN_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/fassert.h>
#include <windows.h>

namespace Falcon
{

/**
   Generic mutex class.

   Directly mapping to the underlying type via inline functions.

   The mutex must be considered as non-reentrant.
*/
class FALCON_DYN_CLASS Mutex
{
   CRITICAL_SECTION m_mtx;

public:
   /** Creates the mutex.
      Will assert on failure.
   */
   inline Mutex()
   {
      //TODO: Remove from inline; this as inline is a mess on windows.
      InitializeCriticalSectionAndSpinCount( &m_mtx, 512 );
   }

   /**
      Destroys the mutex.

      Will assert on failure.
   */
   inline ~Mutex() {
      DeleteCriticalSection( &m_mtx );
   }

   /**
      Locks the mutex.

      Will assert on failure -- but only in debug
   */
   inline void lock()
   {
      EnterCriticalSection( &m_mtx );
   }

   /**
      Unlocks the mutex.

      Will assert on failure -- but only in debug
   */
   inline void unlock()
   {
      LeaveCriticalSection( &m_mtx );
   }

   /**
      Tries to lock the mutex.

      Will assert on failure.
   */
   inline bool trylock()
   {
      return TryEnterCriticalSection( &m_mtx ) == TRUE;
   }

};

/**
   Thread Specific data.

   Directly mapping to the underlying type via inline functions.
*/
class FALCON_DYN_CLASS ThreadSpecific
{
private:
   DWORD m_key;
   void (*m_destructor)(void*);
   ThreadSpecific* m_nextDestructor;

public:
   ThreadSpecific()
   {
      m_key = TlsAlloc();
      m_destructor = 0;
   }

   ThreadSpecific( void (*destructor)(void*) );
   
   virtual ~ThreadSpecific()
   {
      #ifndef NDEBUG
      BOOL res = TlsFree( m_key );
      fassert( res );
      #else
      TlsFree( m_key );
      #endif
   }

   ThreadSpecific* clearAndNext();

   void set( void *value );

   void* get() const
   {
      return TlsGetValue( m_key );
   }
};

/** Performs an atomic thread safe increment. */
inline int32 atomicInc( volatile int32 &data )
{
   volatile LONG* dp = (volatile LONG*) &data;
   return InterlockedIncrement( dp );
}

/** Performs an atomic thread safe decrement. */
inline int32 atomicDec( volatile int32 &data )
{
   volatile LONG* dp = (volatile LONG*) &data;
   return InterlockedDecrement( dp );
}

/**
   Generic event class.

   Directly mapping to the underlying type via inline functions.

   Well, events are definitely not the best way to handle MT things,
   the mutex / POSIX cv / predicate is definitely better (faster, more
   flexible, safer etc), but we're using a set of definite MT  patterns
   in which using MS-WIN style events doesn't make a great difference.

   For low level business (i.e. implementing the script-level Waitable
   system) we're still using the system specific features (multiple
   wait on MS-WIN, condvars on POSIX). This is class is used as
   a middle-level equalizer in simple MT tasks as i.e. signaling
   non-empty queues or generic work-to-be-done flags.
*/
class FALCON_DYN_CLASS Event
{
   HANDLE m_hEvent;

public:
   /** Creates the mutex.
      Will assert on failure.
   */
   inline Event( bool bAutoReset = true, bool initState = false )
   {
      // Second parameter is MANUAL RESET
      m_hEvent = CreateEvent( NULL, bAutoReset ? FALSE : TRUE, initState ? TRUE : FALSE, NULL );
   }

   /**
      Destroys the event.

      Will assert on failure.
   */
   inline ~Event() {
      CloseHandle( m_hEvent );
   }

   /**
      Signals the event.
      Will assert on failure -- but only in debug
   */
   inline void set() { SetEvent( m_hEvent ); }


   /**
      Resets the event.
      Useful only if the event is not auto-reset.
   */
   inline void reset() { ResetEvent( m_hEvent ); }

   /**
      Waits on the given event.

      The wait is not interruptible. If a thread is blocked on this wait, the event must
      be signaled somewhere else to allow it to proceed and check for closure request.

      Falcon script level have better semantics, but this object is meant for fairly basic
      and low-level system related activites.

      If the event is auto-reset, only one waiting thread is woken up, and after the
      wakeup the event is automatically reset.
      \param to The timeout; set to < 0 for infinite timeout, 0 to check without blocking and
         > 0 for a number of MSecs wait.
      \return True if the event was signaled, false otherwise.
   */
   bool wait( int32 to = -1 ) { return WaitForSingleObject( m_hEvent, to >= 0 ? to : INFINITE ) == WAIT_OBJECT_0; }
};


struct SYSTH_DATA {
   HANDLE hThread;
   unsigned nThreadID;
   HANDLE hEvtDetach;
   void *retval;
   
   /** Mutex controlling detachment and termination. */
   CRITICAL_SECTION m_csT;

   /** True when the thread is done and this data is disposeable. */
   bool m_bDone;
   /** Controls joinability and destruction on run exit */
   bool m_bDetached;
   bool m_bJoining;
};


}

#endif

/* end of mt_win.h */
