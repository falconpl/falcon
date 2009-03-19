/*
   FALCON - The Falcon Programming Language.
   FILE: baton_win.cpp

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
#include <falcon/mt_win.h>

namespace Falcon {

typedef struct tag_WIN_BATON_DATA 
{
   HANDLE hEvtIdle;
   HANDLE hEvtUnblocked;
   DWORD nBlockerId;
   CRITICAL_SECTION cs;
} WIN_BATON_DATA;


Baton::Baton( bool bBusy )
{
   m_data = (WIN_BATON_DATA *) memAlloc( sizeof( WIN_BATON_DATA ) );
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   p.nBlockerId = 0;
   // bBusy is reverse with respect to hEvtIdle
   p.hEvtIdle = CreateEvent( NULL, FALSE, bBusy ? FALSE : TRUE , NULL );
   // Unblocker is manual reset (let's everyone go in).
   p.hEvtUnblocked = CreateEvent( NULL, TRUE, TRUE, NULL );
   InitializeCriticalSection( &p.cs );
}


Baton::~Baton()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   
   CloseHandle( p.hEvtIdle );
   CloseHandle( p.hEvtUnblocked );
   DeleteCriticalSection( &p.cs );
   memFree( m_data );
}

void Baton::acquire()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   
   EnterCriticalSection( &p.cs );
   if( p.nBlockerId != GetCurrentThreadId() )
   {
      if ( p.nBlockerId != 0 )
      {
         LeaveCriticalSection( &p.cs );
         // we're sure we hit the block
         onBlockedAcquire();

         // we must wait both for block to be free and idle to be set.
         HANDLE won[] = { p.hEvtIdle, p.hEvtUnblocked };
         DWORD nRes;
         nRes = WaitForMultipleObjects( 2, won, TRUE, INFINITE );
         fassert( nRes == WAIT_OBJECT_0 );
      }
      else 
      {
         // Currently not blocked
         LeaveCriticalSection( &p.cs );
         
         // we must ignore blocking requests incoming now, or we won't be able to signal that we're blocked.
         // In case we receive, the blocker will have to wait for next acquire. Let's say that now,
         // it's already too late to block us.
         DWORD nRes = WaitForSingleObject( p.hEvtIdle, INFINITE );
         fassert( nRes == WAIT_OBJECT_0 );
      }

      // we have acquired the IDLE  event, which is auto reset, so we can proceed.
      return;
   }

   // we are the blocker; must wait only on the idle event.
   LeaveCriticalSection( &p.cs );
   
   DWORD nRes;
   nRes = WaitForSingleObject( p.hEvtIdle, INFINITE );
   fassert( nRes == WAIT_OBJECT_0 );

   EnterCriticalSection( &p.cs );
   p.nBlockerId = 0;
   LeaveCriticalSection( &p.cs );
   
   // no more blocked.
   SetEvent( p.hEvtUnblocked );
}


bool Baton::busy()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
     
   // we must hold the mutex to avoid block to think that the baton is
   // busy, while we're actually keeping it busy here for a short time.
   EnterCriticalSection( &p.cs );
   bool bRet = ! (WaitForSingleObject( p.hEvtIdle, 0 ) == WAIT_OBJECT_0);
   // if it's not busy, we have changed the idle setting; reset it.
   if( ! bRet )
   {
      SetEvent(p.hEvtIdle);
   }
   LeaveCriticalSection( &p.cs );
   return bRet;
}


bool Baton::tryAcquire()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   
   EnterCriticalSection( &p.cs );
   if( p.nBlockerId != GetCurrentThreadId() )
   {
      LeaveCriticalSection( &p.cs );
      
      // we must wait both for block to be free and idle to be set.
      HANDLE won[] = { p.hEvtIdle, p.hEvtUnblocked };
      DWORD nRes;
      nRes = WaitForMultipleObjects( 2, won, TRUE, 0 );
      return (nRes == WAIT_OBJECT_0);
   }

   // we are the blocker; must wait only on the idle event.
   LeaveCriticalSection( &p.cs );
   
   DWORD nRes;
   nRes = WaitForSingleObject( p.hEvtIdle, 0 );
   if ( nRes == WAIT_OBJECT_0 )
   {   
      EnterCriticalSection( &p.cs );
      p.nBlockerId = 0;
      LeaveCriticalSection( &p.cs );
      
      // no more blocked.
      SetEvent( p.hEvtUnblocked );
      return true;
   }
   
   return false;
}


void Baton::release()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   
   // Use sync section to be sure I am not idling the baton while someone
   // is trying to block it.
   EnterCriticalSection( &p.cs );
   SetEvent( p.hEvtIdle );
   LeaveCriticalSection( &p.cs );
}


void Baton::checkBlock()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
      
   EnterCriticalSection( &p.cs );
   if( p.nBlockerId == GetCurrentThreadId() )
   {
      LeaveCriticalSection( &p.cs );

      // I am the blocker; there's nothing we should do now.
      return;
   }
   else {
      if( p.nBlockerId != 0 )
      {
         // Idle in a sync section to avoid idling after blockers block.
         SetEvent( p.hEvtIdle );
         LeaveCriticalSection( &p.cs );
         
         onBlockedAcquire();
         
         // wait for the unblocked event to be on; We just set idle, but we won't steal it
         // as we know for sure that EvtUnblocked is on. If it's not on (i.e. because the
         // blocker has forfaited in the meanwhile) we may re-acquire evtIdle, but that's ok.
         HANDLE won[] = { p.hEvtIdle, p.hEvtUnblocked };
         DWORD nRes;
         nRes = WaitForMultipleObjects( 2, won, TRUE, INFINITE );
         fassert(nRes == WAIT_OBJECT_0);
      }
      else {
         LeaveCriticalSection( &p.cs );
      }
   }

}


bool Baton::block()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   // first, issue a block try
      
   EnterCriticalSection( &p.cs );
   // already blocked? -- we can't do anything.
   if( p.nBlockerId != GetCurrentThreadId() && p.nBlockerId != 0 )
   {
      LeaveCriticalSection( &p.cs );
      return false;
   }
   // Noone can release the baton while we lock the mutex.
   // This gives us the chance to see if the baton is idle.
   if ( WaitForSingleObject( p.hEvtIdle, 0 ) == WAIT_OBJECT_0 )
   {
      // Yes, the baton was idle (we jave just acquired it).
      // This means that noone can reply our block, so we can't perform that.
      // re-idle the baton and go away.
      SetEvent( p.hEvtIdle );
      LeaveCriticalSection( &p.cs );
      return false;
   }
   
   // Ok, we can block. 
   p.nBlockerId = GetCurrentThreadId();
   ResetEvent( p.hEvtUnblocked );
   LeaveCriticalSection( &p.cs );
   
   return true;
}

bool Baton::unblock()
{
   WIN_BATON_DATA &p = *(WIN_BATON_DATA *)m_data;
   EnterCriticalSection( &p.cs );
   if( p.nBlockerId != GetCurrentThreadId() )
   {
      LeaveCriticalSection( &p.cs );
      return false;
   }
   else {
      p.nBlockerId = 0;
      SetEvent( p.hEvtUnblocked );
      LeaveCriticalSection( &p.cs );
   }

   return true;
}

void Baton::onBlockedAcquire()
{
}


}


/* end of baton_win.cpp */
