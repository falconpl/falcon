/*
   FALCON - The Falcon Programming Language.
   FILE: mt_posix.h

   Multithreaded extensions - POSIX specific header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 13:08:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MT_POSIX_H
#define FALCON_MT_POSIX_H

#include <pthread.h>
#include <errno.h>
#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/fassert.h>

namespace Falcon
{

inline void mutex_lock( pthread_mutex_t& mtx )
{
   #ifdef NDEBUG
   pthread_mutex_lock(&mtx);
   #else
   int res = pthread_mutex_lock(&mtx);
   fassert( res != EINVAL );
   fassert( res != EDEADLK );
   fassert( res == 0 );
   #endif
}

inline void mutex_unlock( pthread_mutex_t& mtx )
{
   #ifdef NDEBUG
   pthread_mutex_unlock(&mtx);
   #else
   int res = pthread_mutex_unlock(&mtx);
   fassert( res == 0 );
   #endif
}


inline void cv_wait( pthread_cond_t& cv, pthread_mutex_t& mtx )
{
   #ifdef NDEBUG
   pthread_cond_wait(&cv, &mtx);
   #else
   int res = pthread_cond_wait(&cv, &mtx);
   fassert( res == 0 );
   #endif
}

inline void cv_broadcast( pthread_cond_t& cv )
{
   #ifdef NDEBUG
   pthread_cond_broadcast(&cv);
   #else
   int res = pthread_cond_broadcast(&cv);
   fassert( res == 0 );
   #endif
}

/**
   Generic mutex class.

   Directly mapping to the underlying type via inline functions.

   The mutex must be considered as non-reentrant.
*/
class Mutex
{
   pthread_mutex_t m_mtx;

public:
   /** Creates the mutex.
      Will assert on failure.
   */
   inline Mutex()
   {
      #ifdef NDEBUG
      pthread_mutex_init( &m_mtx, 0 );
      #else
      int result = pthread_mutex_init( &m_mtx, 0 );
      fassert( result == 0 );
      #endif
   }

   /**
      Destroys the mutex.

      Will assert on failure.
   */
   inline ~Mutex() {
      #ifdef NDEBUG
      pthread_mutex_destroy( &m_mtx );
      #else
      int result = pthread_mutex_destroy( &m_mtx );
      fassert( result == 0 );
      #endif
   }

   /**
      Locks the mutex.

      Will assert on failure -- but only in debug
   */
   inline void lock()
   {
      #ifdef NDEBUG
      pthread_mutex_lock( &m_mtx );
      #else
      int result = pthread_mutex_lock( &m_mtx );
      fassert( result != EINVAL );
      fassert( result != EDEADLK );
      #endif
   }

   /**
      Unlocks the mutex.

      Will assert on failure -- but only in debug
   */
   inline void unlock()
   {
      #ifdef NDEBUG
      pthread_mutex_unlock( &m_mtx );
      #else
      int result = pthread_mutex_unlock( &m_mtx );
      fassert( result == 0 );
      #endif
   }

   /**
      Tries to lock the mutex.

      Will assert on failure.
   */
   inline bool trylock()
   {
      int result = pthread_mutex_trylock( &m_mtx );
      if ( result == EBUSY )
         return false;

      #ifndef NDEBUG
      fassert( result == 0 );
      #endif

      return true;
   }

};

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
class Event
{
   pthread_mutex_t m_mtx;
   pthread_cond_t m_cv;
   bool m_bIsSet;
   bool m_bAutoReset;

public:
   /** Creates the mutex.
      Will assert on failure.
   */
   inline Event( bool bAutoReset = true, bool initState = false ):
      m_bIsSet( initState ),
      m_bAutoReset( bAutoReset )
   {
      #ifdef NDEBUG
      pthread_mutex_init( &m_mtx, 0 );
      pthread_cond_init( &m_cv, 0 );
      #else
      int result = pthread_mutex_init( &m_mtx, 0 );
      fassert( result == 0 );
      result = pthread_cond_init( &m_cv, 0 );
      fassert( result == 0 );
      #endif
   }

   /**
      Destroys the event.

      Will assert on failure.
   */
   inline ~Event() {
      #ifdef NDEBUG
      pthread_mutex_destroy( &m_mtx );
      pthread_cond_destroy( &m_cv );
      #else
      int result = pthread_mutex_destroy( &m_mtx );
      fassert( result == 0 );
      result = pthread_cond_destroy( &m_cv );
      fassert( result == 0 );
      #endif
   }

   /**
      Signals the event.
      Will assert on failure -- but only in debug
   */
   void set();


   /**
      Resets the event.
      Useful only if the event is not auto-reset.
   */
   inline void reset()
   {
      #ifdef NDEBUG
      pthread_mutex_lock( &m_mtx );
      m_bIsSet = false;
      pthread_mutex_unlock( &m_mtx );
      #else
      int result = pthread_mutex_lock( &m_mtx );
      fassert( result == 0 );
      m_bIsSet = false;
      result = pthread_mutex_unlock( &m_mtx );
      fassert( result == 0 );
      #endif
   }

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
   bool wait( int32 to = -1 );
};


/**
   Thread Specific data.

   Directly mapping to the underlying type via inline functions.
*/
class ThreadSpecific
{
private:
   pthread_key_t m_key;

public:
   ThreadSpecific()
   {
      pthread_key_create( &m_key, NULL );
   }

   ThreadSpecific( void (*destructor)(void*) );

   virtual ~ThreadSpecific()
   {
      #ifndef NDEBUG
      int value = pthread_key_delete( m_key );
      fassert( value == 0 );
      #else
      pthread_key_delete( m_key );
      #endif
   }

   void set( void *value )
   {
      #ifndef NDEBUG
      int res = pthread_setspecific( m_key, value );
      fassert( res == 0 );
      #else
      pthread_setspecific( m_key, value );
      #endif
   }

   void* get() const
   {
      return pthread_getspecific( m_key );
   }
};


struct SYSTH_DATA {
   pthread_t pth;
   /** Mutex controlling detachment and termination. */
   pthread_mutex_t m_mtxT;
   /** True when the thread is done and this data is disposeable. */
   bool m_bDone;
   /** Controls joinability and destruction on run exit */
   bool m_bDetached; 
   
   int m_lastError;
};

/** Performs an atomic thread safe increment. */
int32 atomicInc( volatile int32 &data );

/** Performs an atomic thread safe decrement. */
int32 atomicDec( volatile int32 &data );

}

#endif

/* end of mt_posix.h */
