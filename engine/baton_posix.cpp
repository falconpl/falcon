/*
   FALCON - The Falcon Programming Language.
   FILE: baton_posix.cpp

   Baton synchronization structure.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 14 Mar 2009 00:03:28 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/memory.h>
#include <falcon/baton.h>
#include <falcon/fassert.h>
#include <falcon/mt_posix.h>

namespace Falcon {

typedef struct tag_POSIX_BATON_DATA
{
   bool m_bBusy;
   bool m_bBlocked;

   pthread_t m_thBlocker;

   pthread_mutex_t m_mtx;
   pthread_cond_t m_cv;
} POSIX_BATON_DATA;


Baton::Baton( bool bBusy )
{
   m_data = (POSIX_BATON_DATA *) memAlloc( sizeof( POSIX_BATON_DATA ) );
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;
   p.m_bBusy = bBusy;
   p.m_bBlocked = false;

#ifdef NDEBUG
   pthread_mutex_init( &p.m_mtx, 0 );
   pthread_cond_init( &p.m_cv, 0 );
#else
   int res = pthread_mutex_init( &p.m_mtx, 0 );
   fassert( res == 0 );
   res = pthread_cond_init( &p.m_cv, 0 );
   fassert( res == 0 );
#endif
}


Baton::~Baton()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

#ifdef NDEBUG
   pthread_mutex_destroy( &p.m_mtx );
   pthread_cond_destroy( &p.m_cv );
#else
   int res = pthread_mutex_destroy( &p.m_mtx );
   fassert( res == 0 );
   res = pthread_cond_destroy( &p.m_cv );
   fassert( res == 0 );
#endif

   memFree( m_data );
}

void Baton::acquire()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   if( p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker ) )
   {
      mutex_unlock( p.m_mtx );
      // no problem if we release the mutex here: it's ok, as the semantic is just that of firing this
      // notify if we're blocked.
      onBlockedAcquire();
      mutex_lock( p.m_mtx );
   }

   while( p.m_bBusy || (p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker ) ) )
   {
      cv_wait( p.m_cv, p.m_mtx );
   }

   p.m_bBusy = true;
   p.m_bBlocked = false;
   mutex_unlock( p.m_mtx );
}


bool Baton::busy()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   bool bbret = p.m_bBusy;
   mutex_unlock( p.m_mtx );
   return bbret;
}

bool Baton::tryAcquire()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   if( p.m_bBusy || (p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker ) ) )
   {
      mutex_unlock( p.m_mtx );
      return false;
   }
   p.m_bBusy = true;
   p.m_bBlocked = false;
   mutex_unlock( p.m_mtx );

   return true;
}

void Baton::release()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   p.m_bBusy = false;
   cv_broadcast( p.m_cv );

   mutex_unlock( p.m_mtx );
}


void Baton::checkBlock()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   p.m_bBusy = false;
   cv_broadcast( p.m_cv );

   bool bb = (p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker ) );

   if ( bb )
   {
      mutex_unlock( p.m_mtx );
      onBlockedAcquire();
      mutex_lock( p.m_mtx );
   }

   while( p.m_bBusy || (p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker ) ) )
   {
      cv_wait( p.m_cv, p.m_mtx );
   }

   p.m_bBusy = true;
   p.m_bBlocked = false;
   mutex_unlock( p.m_mtx );
}


bool Baton::block()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   if ( (!p.m_bBusy) || (p.m_bBlocked && ! pthread_equal( pthread_self(),  p.m_thBlocker )) )
   {
      mutex_unlock( p.m_mtx );
      return false;
   }

   if ( ! p.m_bBlocked )
   {
      p.m_bBlocked = true;
      p.m_thBlocker = pthread_self();
   }
   mutex_unlock( p.m_mtx );

   return true;
}

bool Baton::unblock()
{
   POSIX_BATON_DATA &p = *(POSIX_BATON_DATA *)m_data;

   mutex_lock( p.m_mtx );
   if ( p.m_bBlocked && pthread_equal( pthread_self(),  p.m_thBlocker ) )
   {
      p.m_bBlocked = false;
      cv_broadcast( p.m_cv );
      mutex_unlock( p.m_mtx );

      return true;
   }

   mutex_unlock( p.m_mtx );
   return false;
}

void Baton::onBlockedAcquire()
{
}

}

/* end of baton_posix.cpp */
