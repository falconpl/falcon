/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: systhread_win.cpp

   System dependent MT provider - MS-Windows specific.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Apr 2008 21:32:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent MT provider - MS-Windows specific.
*/

#include "systhread_win.h"

#include <falcon/memory.h>
#include <falcon/vm_sys.h>
#include <falcon/vm_sys_win.h>

#include "mt.h"
#include "systhread.h"
#include <process.h>

static DWORD runningThreadKey = 0;

namespace Falcon {
namespace Sys {

//======================================================
// Mutex
//

void MutexProvider::init( Mutex *mtx, long spinCount )
{
   CRITICAL_SECTION *cs = (CRITICAL_SECTION *) memAlloc( sizeof( CRITICAL_SECTION ) );
   InitializeCriticalSectionAndSpinCount( cs, spinCount );
   mtx->m_sysData = cs;
}


void MutexProvider::destroy( Mutex *mtx )
{
   CRITICAL_SECTION *cs = (CRITICAL_SECTION *) mtx->m_sysData;
   DeleteCriticalSection( cs );
   memFree( cs );
}

bool MutexProvider::trylock( Mutex *mtx )
{
   CRITICAL_SECTION *cs = (CRITICAL_SECTION *) mtx->m_sysData;
   return TryEnterCriticalSection( cs ) == TRUE;
}

void MutexProvider::lock( Mutex *mtx )
{
   CRITICAL_SECTION *cs = (CRITICAL_SECTION *) mtx->m_sysData;
   EnterCriticalSection( cs );
}

void MutexProvider::unlock( Mutex *mtx )
{
   CRITICAL_SECTION *cs = (CRITICAL_SECTION *) mtx->m_sysData;
   LeaveCriticalSection( cs );
}


//======================================================
// Waitable
//

void WaitableProvider::init( Waitable *wo )
{
   wo->m_sysData = CreateEvent( NULL, TRUE, FALSE, NULL );
}

void WaitableProvider::destroy( Waitable *wo )
{
   CloseHandle( ( HANDLE) wo->m_sysData );
}


void WaitableProvider::signal( Waitable *wo )
{
   SetEvent( (HANDLE) wo->m_sysData );
}

void WaitableProvider::broadcast( Waitable *wo )
{
   PulseEvent( (HANDLE) wo->m_sysData );
}

//======================================================
// Thread
//

WIN_TH::WIN_TH( HANDLE canc )
{
   hth = INVALID_HANDLE_VALUE;
   thID = 0;
   lastError = 0;
}

WIN_TH::~WIN_TH()
{
   if( hth != INVALID_HANDLE_VALUE )
      CloseHandle( hth );
}

static unsigned int __stdcall s_threadRunner( void *data )
{
   Thread *runner = (Thread *) data;
   // save the runner as specific for this thread
   TlsSetValue( runningThreadKey, runner );

   void *res = runner->run();
   runner->terminated();
   runner->decref();

   return (unsigned int) res;
}

void ThreadProvider::initSys()
{
   runningThreadKey = TlsAlloc();
}

void ThreadProvider::init( Thread *runner )
{
   WIN_TH *data = new WIN_TH( INVALID_HANDLE_VALUE );
   runner->m_sysData = data;
}

void ThreadProvider::configure( Thread *runner, SystemData &sdt )
{
   WIN_TH *th = (WIN_TH*) runner->m_sysData;
   th->eCancel = sdt.m_sysData->evtInterrupt;
}

bool ThreadProvider::start( Thread *runner, const ThreadParams &params )
{
   WIN_TH *wth = (WIN_TH *) runner->m_sysData;
   
   unsigned int stackSize = params.stackSize();
   unsigned int thid;

   // time to increment the reference count of our thread that is going to run
   runner->incref();

   wth->hth = (HANDLE) _beginthreadex( 
         NULL,
         stackSize,
         s_threadRunner,
         runner,
         0,
         &thid 
      );

   if( wth->hth == INVALID_HANDLE_VALUE )
   {
      runner->decref();
      return false;
   }
   
   wth->thID = thid;

   if ( params.detached() )
   {      
      runner->ThreadStatus::detach();
   }
   return true;
}


void ThreadProvider::interruptWaits( Thread *runner )
{
   // dummy; under windows interrupting the VM is enough
}

Thread *ThreadProvider::getRunningThread()
{
   return (Thread *) TlsGetValue( runningThreadKey );
}

void ThreadProvider::setRunningThread( Thread *th )
{
   // This is a good moment to create the thread key.
   TlsSetValue( runningThreadKey, th );
   th->m_mtx.lock();
   th->m_bStarted = true;
   ((WIN_TH *)th->m_sysData)->thID = GetCurrentThreadId();
   ((WIN_TH *)th->m_sysData)->hth = GetCurrentThread();
   th->m_mtx.unlock();
}


bool ThreadProvider::detach( Thread *runner )
{
   WIN_TH *wth = (WIN_TH *) runner->m_sysData;
   
   if ( ! CloseHandle( wth->hth ) )
   {
      wth->lastError = GetLastError();
      return false;
   }

   return true;
}

void ThreadProvider::destroy( Thread *runner )
{
   WIN_TH *data = (WIN_TH *) runner->m_sysData;
   delete data;
}


uint64 ThreadProvider::getID( const Thread *runner )
{
   WIN_TH *wth = (WIN_TH *) runner->m_sysData;
   return (uint64) wth->thID;
}


uint64 ThreadProvider::getCurrentID()
{
   return (uint64) ::GetCurrentThreadId();
}


bool ThreadProvider::equal( const Thread *th1, const Thread *th2 )
{
   const WIN_TH *pth1 = (WIN_TH *) th1->m_sysData;
   const WIN_TH *pth2 = (WIN_TH *) th2->m_sysData;

   return pth1->thID == pth2->thID;
}


bool ThreadProvider::isCurrentThread( const Thread *runner )
{
   const WIN_TH *pth = (WIN_TH *) runner->m_sysData;

   return pth->thID == ::GetCurrentThreadId();
}


int ThreadProvider::waitForObjects( const Thread *runner, int32 count, Waitable **objects, int64 time )
{
   WIN_TH *data = (WIN_TH *) runner->m_sysData;

   // first, let's see if we have some data ready
   // in case of no wait, just check for availability

   for ( int32 i = 0; i < count; i++ )
   {
      // try-acquire semantic.
      if ( objects[i]->acquire() )
      {
         // eventually clear signalation on this object
         WaitForSingleObject( (HANDLE) objects[i]->m_sysData, 0 );
         return i;
      }
   }

   // noway.
   if ( time == 0 )
      return -1;

   HANDLE waited[ MAX_WAITER_OBJECTS ];
   for ( int wid = 0; wid < count && wid < MAX_WAITER_OBJECTS; wid ++ )
      waited[wid] = (HANDLE) objects[wid]->m_sysData;
   
   int cancHandle;
   if ( data->eCancel != INVALID_HANDLE_VALUE )
   {
      waited[count] = data->eCancel;
      cancHandle = 1;
   }
   else
      cancHandle = 0;

   // TODO: Add cancellation event

   DWORD now;
   DWORD targetTime;

   if( time < 0 )
   {
      now = 0;
      targetTime = INFINITE;
   }
   else 
   {
      now = ::GetTickCount();
      targetTime = now + (DWORD)(time/1000);
   }

   // now wait till we can acquire something.
   int acquired = -1;
   while( true )
   {
      DWORD res = WaitForMultipleObjects( count+cancHandle, waited, FALSE, targetTime - now );
      if ( res == WAIT_TIMEOUT )
      {
         // nothing to do...
         return -1;
      }
      else if ( res == WAIT_OBJECT_0 + count )
      {
         // Cancelled
         return -2;
      }
      
      // try to acquire the signaled item.
      acquired = res - WAIT_OBJECT_0;
      if ( acquired < 0 || acquired >= count || ! objects[acquired]->acquire() )
      {
         // try again. If we have a target time, update wait
         if ( time > 0 )
         {
            now = ::GetTickCount();
            if ( now >= targetTime )
            {
               // we timed out
               return -1;
            }
            // otherwise, continue
         }
      }
      else {
         // we acquired the object
         // Reset the event; just to prevent other wakeups, even if we know that some went.
         ResetEvent( (HANDLE) objects[acquired]->m_sysData );
         break; // so we can return outside the loop
      }
   }
   
   return acquired;
}

}
}

/* end of systhread_win.cpp */
