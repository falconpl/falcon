/*
   FALCON - The Falcon Programming Language.
   FILE: mt_win.cpp

   Multithreaded extensions - MS-Windows specific.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 17 Jan 2009 17:06:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/mt.h>
#include <falcon/memory.h>
#include <process.h>

namespace Falcon
{
//==================================================================================
// System threads. 
//

SysThread::~SysThread() 
{
   CloseHandle( m_sysdata->hEvtDetach );
   memFree( m_sysdata );
}

SysThread::SysThread( Runnable* r ):
   m_runnable( r )
{
   m_sysdata = ( struct SYSTH_DATA* ) memAlloc( sizeof( struct SYSTH_DATA ) );
   // The event isactually a barrier.
   m_sysdata->hEvtDetach = CreateEvent( 0, TRUE, FALSE, 0 );
   m_sysdata->retval = 0;
}

void SysThread::attachToCurrent()
{
   m_sysdata->hThread = GetCurrentThread();
   m_sysdata->nThreadID = GetCurrentThreadId();
}

extern "C" {
   static unsigned int __stdcall run_a_thread( void *data )
   {
      return (unsigned int) SysThread::RunAThread( data );
   }
}

void* SysThread::RunAThread( void *data )
{
   SysThread* sth = (SysThread*) data;
   void* retval = sth->run();
   // If the thread is detached, we got to destroy it.
   if ( WaitForSingleObject( sth->sysdata()->hEvtDetach, 0 ) == WAIT_OBJECT_0 )
      delete sth;
   else
      sth->sysdata()->retval = retval;
  
   return retval;
}


bool SysThread::start( const ThreadParams &params )
{
   m_sysdata->hThread = (HANDLE) _beginthreadex( 0, params.stackSize(), &run_a_thread, this, 0, &m_sysdata->nThreadID );
   if ( m_sysdata->hThread == INVALID_HANDLE_VALUE )
      return false;
      
   return true;
}

void SysThread::detach()
{
   SetEvent( m_sysdata->hEvtDetach );
}
   
bool SysThread::join( void* &result )
{
   HANDLE hs[] = { m_sysdata->hEvtDetach, m_sysdata->hThread };
   DWORD wres = WaitForMultipleObjects( 2, hs, FALSE, INFINITE );
   
   if ( wres == WAIT_OBJECT_0 )
   {
      // Ok, we joined the thread.
      result = m_sysdata->retval;
      delete this;
      return true;
   }
   
   return false;
}


uint64 SysThread::getID()
{
   return (uint64) m_sysdata->nThreadID;
}
   
uint64 SysThread::getCurrentID()
{
   return (uint64) GetCurrentThreadId();
}
   
bool SysThread::isCurrentThread()
{
   return GetCurrentThreadId() == m_sysdata->nThreadID;
}
   
bool SysThread::equal( const SysThread *th1 ) const
{
   return m_sysdata->nThreadID == th1->m_sysdata->nThreadID;
}

void *SysThread::run()
{
   fassert( m_runnable !=  0 );
   return m_runnable->run();
}

}

/* end of mt_win.cpp */
