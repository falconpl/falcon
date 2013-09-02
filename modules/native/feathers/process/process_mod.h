/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.h

   System dependent module specifications for process module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Sep 2013 19:10:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_PROCESS_MOD_H
#define FALCON_FEATHERS_PROCESS_MOD_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/mt.h>
#include <falcon/shared.h>

#include "process.h"

namespace Falcon {
class Stream;
class VMContext;
class Class;

namespace Mod {

/** Class representing an attached (child) process. */
class Process: public Shared
{

public:
   Process( VMContext* ctx, const Class* handler );
   virtual ~Process();

   static const int SINK_INPUT = 0x1;
   static const int SINK_OUTPUT = 0x2;
   static const int SINK_AUX = 0x4;
   static const int MERGE_AUX = 0x8;
   static const int BACKGROUND = 0x10;
   static const int USE_SHELL = 0x20;
   static const int USE_PATH = 0x40;

   Stream* inputStream() const;
   Stream* outputStream() const;
   Stream* errorStream() const;

   void open( const String& args, int openMode, bool async = true );

   /** Asynchronously (try to) terminate the process.
    Returns false if the process is not open.
    Throws on system error.
    * */
   bool terminate( bool severe = false );

   /** Returns true if the process is terminated. */
   bool exitValue( int& value ) const;

   void waitTermination();
   void close();

   /** Command started by this process, if any */
   const String& cmd() const { return m_cmd; }

   int64 pid() const;
   int32 exitValue() const { return m_exitval; }

   // Redefine consume-signal as processes are always signaled when finished.
   int32 consumeSignal( VMContext*, int32 count );
   // Redefine consume-signal as processes are always signaled when finished.
   int lockedConsumeSignal( VMContext*, int count );

private:
   bool m_bOpen;
   bool m_done;
   int m_exitval;
   // mutex locking all the functionalities.
   mutable Mutex m_mtx;
   // waiter therad
   SysThread* m_twThread;

   Stream* m_stdIn;
   Stream* m_stdOut;
   Stream* m_stdErr;

   String m_cmd;

   class TermWaiter: public Runnable
   {
   public:
      TermWaiter( Process* owner );
      virtual ~TermWaiter();
      virtual void* run();

   private:
      Process* m_process;
   };

   TermWaiter m_tw;
   friend class TermWaiter;

   // Class holding the system-specific data.
   class Private;
   Private *_p;


   void sys_init();
   void sys_destroy();

   // system-level open
   void sys_open( const String& cmd, int openParams );
   void sys_wait();
   void sys_close();
};


/** Class used to enumerate the running processes. */
class ProcessEnum
{
   void *m_sysdata;

public:
   ProcessEnum();
   virtual ~ProcessEnum();

   /** Get next entry in the enum.
      \return -1 on error, 0 on done, 1 on next available.
   */
   int next( String &name, uint64 &pid, uint64 &ppid, String &cmdLine );
   bool close();
};

uint64 processId();
bool processKill( uint64 id );
bool processTerminate( uint64 id );


}} // ns Falcon::Mod

#endif

/* end of process_mod.h */
