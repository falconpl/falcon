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
namespace Sys {

typedef int (*threadRoutine)( void *param );
class ThreadParams;
class Thread;
class Waitable;
class SystemData;

/** System dependent thread provider.
   This class is actually a collection of static routines that are
   declared as "friend" of the base class Thread.
*/
class ThreadProvider
{
public:
   static void initSys();
   static void init( Thread *runner );
   static void configure( Thread *runner, SystemData &sdt );
   static bool start( Thread *runner, const ThreadParams &params );
   static bool stop( Thread *runner );
   static void interruptWaits( Thread *th );
   static bool detach( Thread *runner );
   static void destroy( Thread *runner );
   static uint64 getID( const Thread *runner );
   static uint64 getCurrentID();
   static bool equal( const Thread *th1, const Thread *th2 );
   static bool isCurrentThread( const Thread *runner );
   static int waitForObjects( const Thread *runner, int32 count, Waitable **objects, int64 time );
   static void wakeUp( Thread *runner, Waitable *object );
   static Thread *getRunningThread();
   static void setRunningThread( Thread *th );
};


class Mutex;

/** System dependent mutex provider.
   This class is actually a collection of static routines that are
   declared as "friend" of the base class Mutex.
*/
class MutexProvider
{
public:
   static void init( Mutex *mtx, long spincount );
   static void destroy( Mutex *mtx );

   static bool trylock( Mutex *mtx );
   static void lock( Mutex *mtx );
   static void unlock( Mutex *mtx );
};


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
};

}
}

#endif

/* end of systhread.h */
