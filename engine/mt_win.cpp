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
   DeleteCriticalSection( &m_sysdata->m_csT );
   memFree( m_sysdata );
}

SysThread::SysThread( Runnable* r ):
   m_runnable( r )
{
   m_sysdata = ( struct SYSTH_DATA* ) memAlloc( sizeof( struct SYSTH_DATA ) );
   // The event isactually a barrier.
   m_sysdata->hEvtDetach = CreateEvent( 0, TRUE, FALSE, 0 );
   m_sysdata->retval = 0;

   m_sysdata->m_bDone = false;
   m_sysdata->m_bDetached = false;
   m_sysdata->m_bJoining = false;
   InitializeCriticalSection( &m_sysdata->m_csT );
}

void SysThread::disengage()
{
   delete this;
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
   return sth->run();
}


bool SysThread::start( const ThreadParams &params )
{
   m_sysdata->hThread = (HANDLE) _beginthreadex( 0, params.stackSize(), &run_a_thread, this, 0, &m_sysdata->nThreadID );
   if ( m_sysdata->hThread == INVALID_HANDLE_VALUE )
      return false;
   
   if ( params.detached() )
      detach();

   return true;
}

void SysThread::detach()
{
   EnterCriticalSection( &m_sysdata->m_csT );
   if( m_sysdata->m_bDone )
   {
      // if we're done, either we or the joiner must destroy us.
      if ( m_sysdata->m_bJoining )
      {
         m_sysdata->m_bDetached = true;
         SetEvent( m_sysdata->hEvtDetach );
         LeaveCriticalSection( &m_sysdata->m_csT );
      }
      else {
         // this prevents joiners to succed while we destroy ourself
         m_sysdata->m_bJoining = true;
         LeaveCriticalSection( &m_sysdata->m_csT );
         delete this;
      }
   }
   else {
      m_sysdata->m_bDetached = true;
      SetEvent( m_sysdata->hEvtDetach );
      LeaveCriticalSection( &m_sysdata->m_csT );
   }
}
   
bool SysThread::join( void* &result )
{
   // ensure just one thread can join.
   EnterCriticalSection( &m_sysdata->m_csT );
   if ( m_sysdata->m_bJoining || m_sysdata->m_bDetached )
   { 
      LeaveCriticalSection( &m_sysdata->m_csT );
      return false;
   }
   else {
      m_sysdata->m_bJoining = true;
      LeaveCriticalSection( &m_sysdata->m_csT );
   }
   
   HANDLE hs[] = { m_sysdata->hEvtDetach, m_sysdata->hThread };
   DWORD wres = WaitForMultipleObjects( 2, hs, FALSE, INFINITE );
   
   if ( wres == WAIT_OBJECT_0 )
   {
      // The thread was detached -- if it's also done, we must destroy it.
      EnterCriticalSection( &m_sysdata->m_csT );
      if( m_sysdata->m_bDone )
      {
         LeaveCriticalSection( &m_sysdata->m_csT );
         delete this;
         return false;
      }
      
      m_sysdata->m_bJoining = false;
      LeaveCriticalSection( &m_sysdata->m_csT );
      return false;  // can't join anymore.
   }
   else if ( wres == WAIT_OBJECT_0 + 1 )
   {
      // Ok, we joined the thread -- and it terminated.
      result = m_sysdata->retval;
      delete this;
      return true;
   }
   
   // wait failed.
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
   m_sysdata->retval = m_runnable->run();
   
   // if we're detached and not joined, we must destroy ourself.
   EnterCriticalSection( &m_sysdata->m_csT );
   if( m_sysdata->m_bDetached || ! m_sysdata->m_bJoining )
   {
      LeaveCriticalSection( &m_sysdata->m_csT );
      delete this;
      return 0;
   }
   else {
      // otherwise, just let joiners or detachers to the dirty job.
      m_sysdata->m_bDone = true;
      LeaveCriticalSection( &m_sysdata->m_csT );
   }
   
   return m_sysdata->retval;
}

}

/* end of mt_win.cpp */
