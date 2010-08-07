/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: systhread.h

   System dependent MT provider.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Apr 2008 21:32:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent MT provider.
*/

#ifndef FLC_SYSTHREAD_H
#define FLC_SYSTHREAD_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <systhread.h>

#ifdef MAXIMUM_WAIT_OBJECTS
   #define MAX_WAITER_OBJECTS (MAXIMUM_WAIT_OBJECTS-1)
#else
   #define MAX_WAITER_OBJECTS    32
#endif

namespace Falcon {
namespace Ext {

class ThreadImpl;
class Waitable;

class WaitableProvider
{
public:
   static void init( Waitable *wo );
   static void destroy( Waitable *wo );
   static void signal( Waitable *wo );
   static void broadcast( Waitable *wo );

   static void lock( Waitable *wo );
   static bool acquireInternal( Waitable *wo );
   static void unlock( Waitable *wo );

   static void interruptWaits( ThreadImpl *runner );
   static int waitForObjects( const ThreadImpl *runner, int32 count, Waitable **objects, int64 time );
};

/** Return threading module system specific data. */
void* createSysData();

/** Dispose threading module system specific data. */
void disposeSysData( void *data );

}
}

#endif

/* end of systhread.h */
