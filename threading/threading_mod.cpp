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

#include "threading_mod.h"

namespace Falcon {
namespace Ext {


//========================================================
// Thread carrier
//

ThreadCarrier::ThreadCarrier( VMRunnerThread *t ):
   WaitableCarrier( t )
{
}

ThreadCarrier::ThreadCarrier( const ThreadCarrier &other ):
   WaitableCarrier( other )
{
}

ThreadCarrier::~ThreadCarrier()
{}


FalconData *ThreadCarrier::clone() const
{
   return new ThreadCarrier( *this );
}

//========================================================
// VMRunner thread
//

VMRunnerThread::VMRunnerThread():
   m_vm( new VMachine ),
   m_lastError( 0 )
{
   // remove the error handler
   m_vm->launchAtLink( false );

   // configure through system data
   Sys::ThreadProvider::configure( this, m_vm->systemData() );
}

VMRunnerThread::VMRunnerThread( VMachine *vm ):
   m_vm( vm ),
   m_lastError( 0 )
{
   m_vm->incref();
   m_vm->launchAtLink( false );
   m_bStarted = true; // an adopted VM is running.
}

VMRunnerThread::~VMRunnerThread()
{
   m_vm->decref();
}

void VMRunnerThread::prepareThreadInstance( const Item &instance, const Item &method )
{
   // create the instance of the method for faster identify
   // The caller should have already certified that the instance has a "run" method
   fassert( method.isCallable() );
   m_threadInstance = instance;
   m_method = method;
}


void *VMRunnerThread::run()
{
   // hold a lock to the item, as it cannot be taken in the vm.
   GarbageLock *tiLock = m_vm->lock( m_threadInstance );
   GarbageLock *mthLock = m_vm->lock( m_method );

   // Perform the call.
   try {
      m_vm->callItem( m_method, 0 );
   }
   catch( Error* err )
   {
      err->incref();
      m_lastError = err;
   }

   // unlock the threads objects
   m_vm->unlock( tiLock );
   m_vm->unlock( mthLock );
   return 0;
}

//=========================================================
// WaitableCarrier
//

WaitableCarrier::WaitableCarrier( ::Falcon::Sys::Waitable *t ):
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
