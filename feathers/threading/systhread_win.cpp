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

#include "waitable.h"
#include "systhread.h"
#include "threading_mod.h"
#include <process.h>

static DWORD runningThreadKey = 0;

namespace Falcon {
namespace Ext {

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

WIN_THI_DATA::WIN_THI_DATA()
{
   hth = INVALID_HANDLE_VALUE;
   thID = 0;
   lastError = 0;
}

WIN_THI_DATA::~WIN_THI_DATA()
{
   if( hth != INVALID_HANDLE_VALUE )
      CloseHandle( hth );
}


void WaitableProvider::interruptWaits( ThreadImpl *runner )
{
   // dummy; under windows interrupting the VM is enough
}

int WaitableProvider::waitForObjects( const ThreadImpl *runner, int32 count, Waitable **objects, int64 time )
{
   WIN_THI_DATA *data = (WIN_THI_DATA *) runner->sysData();
   const ::Falcon::Sys::SystemData &sd = runner->vm().systemData();

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
   Sys::VM_SYS_DATA* vmsd = sd.m_sysData;
   if ( vmsd->evtInterrupt != INVALID_HANDLE_VALUE )
   {
      waited[count] = vmsd->evtInterrupt;
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


void* createSysData() {
   return new WIN_THI_DATA;
}


void disposeSysData( void *data )
{
   if ( data != 0 )
   {
      WIN_THI_DATA* ptr = (WIN_THI_DATA*) data;
      delete ptr;
   }
}

}
}

/* end of systhread_win.cpp */
