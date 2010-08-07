/*
   FALCON - The Falcon Programming Language.
   FILE: threading_mod.cpp

   Threading module binding extensions - internal implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Apr 2008 23:26:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Threading module binding extensions - internal implementation.
*/

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/coreobject.h>
#include <falcon/vm.h>
#include <falcon/garbagelock.h>

#include "threading_mod.h"

namespace Falcon {
namespace Ext {

static void vmthread_killer( void *vmt ) 
{
   ThreadImpl *th = static_cast<ThreadImpl*>( vmt );
   th->decref();
}

static ThreadSpecific m_curthread( vmthread_killer );

void setRunningThread( ThreadImpl* th )
{
   ThreadImpl *old = static_cast<ThreadImpl*>( m_curthread.get() );
   if ( old != 0 )
      old->decref();
   
   if ( th != 0 )
      th->incref();

   m_curthread.set( th );
}

ThreadImpl* getRunningThread()
{
   return static_cast<ThreadImpl*>( m_curthread.get() );
}

//========================================================
// Thread carrier
//

ThreadCarrier::ThreadCarrier( ThreadImpl *t ):
   m_thi( t )
{
}

ThreadCarrier::ThreadCarrier( const ThreadCarrier &other ):
   m_thi( other.m_thi )
{
   m_thi->incref();
}

ThreadCarrier::~ThreadCarrier()
{
   m_thi->decref();
}


FalconData *ThreadCarrier::clone() const
{
   return new ThreadCarrier( *this );
}

//========================================================
// VMRunner thread
//

static int s_threadId = 0;

ThreadImpl::ThreadImpl():
   m_nRefCount(1),
   m_sth(0),
   m_vm( new VMachine ),
   m_lastError( 0 ),
   m_id( atomicInc( s_threadId ) )
{
   m_sysData = createSysData();
}

ThreadImpl::ThreadImpl( const String &name ):
   m_nRefCount(1),
   m_sth(0),
   m_vm( new VMachine ),
   m_lastError( 0 ),
   m_id( atomicInc( s_threadId ) ),
   m_name( name )
{
   m_sysData = createSysData();
}


ThreadImpl::ThreadImpl( VMachine *vm ):
   m_nRefCount(1),
   m_vm( vm ),
   m_lastError( 0 ),
   m_id( atomicInc( s_threadId ) )
{
   m_vm->incref();
   m_thstatus.startable(); // an adopted VM is running.
   m_sth = new SysThread;
   m_sth->attachToCurrent();
   
   m_sysData = createSysData();
}

ThreadImpl::~ThreadImpl()
{
   m_vm->decref();
   // don't delete sth; it has been disposed by detach or join.
   if ( m_lastError != 0 )
      m_lastError->decref();
      
   disposeSysData( m_sysData );
   
   if ( m_sth != 0 )
   {
      void *data;
      // ok even if detached.
      m_sth->join( data );
   }
}

void ThreadImpl::incref()
{
   atomicInc( m_nRefCount );
}

void ThreadImpl::decref()
{
   int count = atomicDec( m_nRefCount );
   
   if ( count == 0 )
      delete this;
}


void ThreadImpl::prepareThreadInstance( const Item &instance, const Item &method )
{
   // create the instance of the method for faster identify
   // The caller should have already certified that the instance has a "run" method
   fassert( method.isCallable() );
   m_threadInstance = instance;
   m_method = method;
}


void *ThreadImpl::run()
{
   m_vm->incref();
   setRunningThread( this );
   
   // hold a lock to the item, as it cannot be taken in the vm.
   GarbageLock tiLock( m_threadInstance );
   GarbageLock mthLock( m_method );

   // Perform the call.
   try {
      m_vm->callItem( m_method, 0 );
      m_lastError = 0;
   }
   catch( Error* err )
   {
      m_lastError = err;
   }

   m_vm->finalize();  // and we won't use it anymore
   m_thstatus.terminated();

   return 0;
}


bool ThreadImpl::start( const ThreadParams &params )
{
   // never call this before startable... so.
   fassert( m_sth == 0 );
   m_sth = new SysThread(this);
   return m_sth->start( params );
}

bool ThreadImpl::detach()
{
   if( m_thstatus.detach() )
   {
      m_sth->detach();
      m_sth = 0;
      return true;
   }
   
   return false;
}

//=========================================================
// WaitableCarrier
//

WaitableCarrier::WaitableCarrier( Waitable *t ):
   m_wto( t )
{
   t->incref();
}

WaitableCarrier::WaitableCarrier( const WaitableCarrier &other )
{
   m_wto = other.m_wto;
   m_wto->incref();
}

WaitableCarrier::~WaitableCarrier()
{
   m_wto->decref();
}

FalconData *WaitableCarrier::clone() const
{
   return new WaitableCarrier( *this );
}


}
}
/* end of threading_mod.cpp */
