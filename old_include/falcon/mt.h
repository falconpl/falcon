/*
   FALCON - The Falcon Programming Language.
   FILE: mt.h

   Multithreaded extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 13:08:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MT_H
#define FALCON_MT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

#ifdef FALCON_SYSTEM_WIN
#include <falcon/mt_win.h>
#else
#include <falcon/mt_posix.h>
#endif

namespace Falcon {

/** Runnable base class.
   When the abstraction of a thread is quite good to represent the final object
   to be run, this pure interface can be passed to a thread that will run it.
*/

class FALCON_DYN_CLASS Runnable
{
public:
   virtual ~Runnable() {};
   virtual void* run() = 0;
};

/** Thread creation parameters. */
class FALCON_DYN_CLASS ThreadParams: public BaseAlloc
{
   uint32 m_stackSize;
   bool m_bDetached;

public:
   ThreadParams():
      m_stackSize(0),
      m_bDetached( false )
   {}

   ThreadParams &stackSize( uint32 size ) { m_stackSize = size; return *this; }
   ThreadParams &detached( bool setting ) { m_bDetached = setting; return *this; }

   uint32 stackSize() const { return m_stackSize; }
   bool detached() const { return m_bDetached; }
};

struct SYSTH_DATA;

/** System Thread class.
   Can be used both to run a Runnable class instance, or to run its own
   virtual run method.
   
   This class is called SysThread to distinguish it from the higher level
   Thread. This class just encapsulates the system threading, wheres
   Thread gives more multithreading programming high level support,
   as interrupt management and multiple object wait.
   
   Notice that we don't have a stop() or cancel() or interrupt() method in
   this class. This is because all stop requests are handled at higher level,
   by the threads interested in VM operations (so that they can be handled
   i.e. by the scripts). MS-Windows doesn't provide a cancelation protocol,
   while POSIX cancelation system doesn't allow for handling of the cancelation
   event (the semantic is such that cancelation of I/O blocking operations must
   be honored ASAP, with at least a small control on cleanup). Both ways to do
   the thing (i.e. none or too much) aren't good for the code we're writing
   above this class, so we just drop this feature and re-implement cancelation
   requests at higher level, preventing - controlling blocking I/O.
   
*/
class FALCON_DYN_CLASS SysThread: public BaseAlloc
{
   struct SYSTH_DATA* m_sysdata;
   
protected:
   Runnable* m_runnable;
   virtual ~SysThread();
   
public:
   SysThread( Runnable* r = 0 );
   
   /** Makes this object to represent currently running system thread. 
      This must be called after the constructor and before start().
   */
   void attachToCurrent();
   
   /** Launches a new instance of this thread.
      
      Only one start can be performed for a thread. Trying to execute start more
      than once will fail.
   */
   bool start( const ThreadParams &params );
   
   /** Launches a new instance of this thread with default parameters.
      
      Only one start can be performed for a thread. Trying to execute start more
      than once will fail.
   */
   bool start() { return start( ThreadParams() ); }
   
   /** Detach this thread.
      After this call, the thread will take care of destroying itself at termination.
      Joining threads will receive an unjoinable exception.
      
      Notice that if the run() return value is dynamically allocate, it will be leaked,
      so this method should be called only for threads not returning any value.
      
      After this call, this instance must be considered invalid.
      
      \note A SysThread must terminate either because a detach or with a join. The destroyer
      cannot be called directly (this means you can't create a SysThread instance
      in the stack).
   */
   void detach();
   
   /** Join this thread.
      Wait for this thread to terminate, and return the value returned by run(). After
      join has returned, this thread cannot be used anymore (it is virtually destroyed).
      
      \note A SysThread must terminate either because a detach or with a join. The destroyer
      cannot be called directly (this means you can't create a SysThread instance
      in the stack).
      
      \param result The output of the run() method of this thread.
      \return false if the thread is not joinable, true if it has been joined.
   */
   bool join( void* &result );
   
   /** Returns the current thread ID.
      On POSIX systems, it returns the value of an unique value associated with each thread;
      on MS-Windows systems returns the system Thread ID.
   */
   uint64 getID();
   
   /** Returns the thread ID of the running thread.
      On POSIX systems, it returns the value of an unique value associated with each thread;
      on MS-Windows systems returns the system Thread ID.
   */
   static uint64 getCurrentID();
   
   /** Returns true if the thread is currently running. */
   bool isCurrentThread();
   
   /** Returns true if two threads represents the same low level threads */
   bool equal( const SysThread *th1 ) const;

   static bool equal( const SysThread *th1, const SysThread *th2 ) {
      return th1->equal( th2 );
   }
   
   /** Run method.
      The base class method just runs the runnable's run(). 
      It asserts( or crashes) if the runnable has not been set in the first place.
      
      Subclasses are free to set everything they want as the run method.
   */
   virtual void* run();
   
   SYSTH_DATA *sysdata() const { return m_sysdata; }
   
   static void *RunAThread(void *);
   
   /** Dispose of this object without modifying the underlying system data. */
   void disengage();
};

}

#endif

/* end of mt.h */
