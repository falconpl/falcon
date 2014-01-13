/*
   FALCON - The Falcon Programming Language.
   FILE: vm.h

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_VM_H
#define FALCON_VM_H

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/codeframe.h>
#include <falcon/callframe.h>
#include <falcon/vmcontext.h>
#include <falcon/string.h>
#include <falcon/syncqueue.h>
#include <falcon/contextmanager.h>
#include <falcon/scheduler.h>
#include <falcon/pdata.h>

#include <falcon/mersennetwister.h>

#define FALCON_VM_DFAULT_CHECK_LOOPS 5000

namespace Falcon {

class Error;
class LocationInfo;
class Stream;
class TextReader;
class TextWriter;
class Transcoder;
class Module;
class Symbol;
class ModSpace;
class ModLoader;
class Function;
class MessageQueue;

/** The Falcon virtual machine.
*/
class FALCON_DYN_CLASS VMachine
{

public:

   /** Creates the virtual machine.
    \param stdIn Standard input stream used by the VM; if 0, will take a duplicate
    of the process standard input stream.
    \param stdOut Standard output stream used by the VM; if 0, will take a duplicate
    of the process standard output stream.
    \param stdErr Standard input stream used by the VM; if 0, will take a duplicate
    of the process standard error stream.

    The streams passed to the constructor are property of the virtual machine;
    they get destroyed (and closed) with it.

    If the script that should be run by this virtual machine is seen as the
    foreground process, the creator of this virtual machine should pass standard
    stream instances not duplicating the standard streams, so that the script
    can close the VM streams and that behavior is reflected to the process streams.

    \note The streams given to the virtual machine are NOT increffed. If you want to keep them
    beyond VM Destruction, they must receive an extra reference before calling this constructor.
    */
   VMachine( Stream* stdIn = 0, Stream* stdOut=0, Stream* stdErr = 0 );
   virtual ~VMachine();

   /** Assigns a context to this virtual machine.

       The context is put in the scheduler for immediate execution,
       and a new ID is assigned to it.

       The context should not be assigned to another VM, or the process
       will die.

       \param ctx The context being assigned.
    */
   void addContextGroup( ContextGroup* grp );


   //=========================================================
   // Backward compatiblity
   //=========================================================

   /** Sets a value in the A register of the current context.
    \param v The return value.
    \deprecated  Kept for compatibility with engine 0.9.x

    */
   void retval( const Item& v );

   /** Access the current context accumulator.
     \deprecated  Kept for compatibility with engine 0.9.x

    */
   const Item& regA() const;
   /** Access the current context accumulator.
    \deprecated  Kept for compatibility with engine 0.9.x
    */
   Item& regA();

   /** Access the current "self" item.
        \deprecated  Kept for compatibility with engine 0.9.x

    */
   const Item& self() const;
   /** Access the current context "self" item.
     \deprecated  Kept for compatibility with engine 0.9.x

    */
   Item& self();


   //=========================================================
   // VM Streams
   //=========================================================

   /** Changes the standard input stream.
    Previously owned standard input stream is closed and destroyed.
    */
   void stdIn( Stream* s );

   /** Changes the standard output stream.
    Previously owned standard output stream is closed and destroyed.
    */
   void stdOut( Stream* s );

   /** Changes the standard error stream.
    Previously owned standard error stream is closed and destroyed.
    */
   void stdErr( Stream* s );

   /** Returns current standard input stream.
    \return A valid standard input stream, owned by the VM.
    If needed elsewhere, the stream must be cloned().
   */
   inline Stream* stdIn() const { return m_stdIn; }

   /** Returns current standard input stream.
    \return A valid standard output stream, owned by the VM.
    If needed elsewhere, the stream must be cloned().
   */
   inline Stream* stdOut() const { return m_stdOut; }

   /** Returns current standard error stream.
    \return A valid standard error stream, owned by the VM.
    If needed elsewhere, the stream must be cloned().
   */
   inline Stream* stdErr() const { return m_stdErr; }

   /** Sets the standard encoding of streams.
    \param Encoding name as ISO encoding name.
    \return false if the given encoding is currently not served by the engine.

    The VM offers an utility TextReader to read from its standard input
    and TextWriter wrappers to write to its standard output and error
    streams.

    Applications are not required to use this readers/writers, but
    the standard library text based I/O functions use them.

    The text readers/writers are initialized to the system encoding, if it is
    possible to detect it and if the Transcoder is present in the Engine at the
    moment of VM creation.

    It is possible to change the transcoder used by the standard library text
    I/O routines anytime through the setStdEncoding() method. It is also possible
    to provide a custom transcoder, that can be disposed by the VM.

    \note The text writers used by the VM set their CRLF automatic transcoding
    option according to the host system, and with automatic flush at end-of-line;
    to change this setup, act directly on the TextWriters.

    */
   bool setStdEncoding( const String& name );

    /** Sets the standard encoding of streams using a custom transcoder.
     \param ts A transcoder instance.
     \param bOwn If true, the transcoder will be disposed by the VM at destruction.
     \see bool setStdEncoding( const String& name )
     */
   void setStdEncoding( Transcoder* ts );

   Transcoder* getStdEncoding() const { return m_stdCoder; }

   /** Returns the TextReader accessing the standard input stream.
    \return A text reder.
    \see setStdEncoding
    */
   inline TextReader* textIn() const { return m_textIn; }

   /** Returns the TextReader accessing the standard input stream.
    \return A text reder.
    \see setStdEncoding
    \note The text writers used by the VM set their CRLF automatic transcoding
    option according to the host system, and with automatic flush at end-of-line;
    to change this setup, act directly on the TextWriters.
    */
   inline TextWriter* textOut() const { return m_textOut; }

   /** Returns the TextReader accessing the standard input stream.
    \return A text reder.
    \see setStdEncoding
    \note The text writers used by the VM set their CRLF automatic transcoding
    option according to the host system, and with automatic flush at end-of-line;
    to change this setup, act directly on the TextWriters.
    */
   inline TextWriter* textErr() const { return m_textErr; }

   /** Cleanly terminates the virtual machine.
    *
    The scheduler and all the processors are sent a request
    to terminate any operation as soon as possible.
    */
   void quit();

   /** Creates a new non configured process process.
    The new process is not started until explicitly requested.

    The returned process has 1 extra reference count for the caller;
    if the caller is not interested in the process, it should dereference
    it before leaving the invoking function.

    The created process is not ready to be run. It
      \note The ID of the returned process is atomically increased at each
      invocation. ID of terminated processes are not recycled.
    */
   Process* createProcess();

   /** Gets a VM process by id, or 0 if not found.
       \return A valid process entity, or 0 if the process with that
       ID is currently terminated or not available.
   */
   Process* getProcessByID( int32 pid );

   /**
    * Adds a process created elsewhere to this VM.
    * \param p the Process to be added.
    * \param bLaunch if true, make immediately runnable.
    *
    * This is usually done automatically by Process::launch(), if needed.
    */
   void addProcess( Process* p, bool bLaunch );

   /** Changes the active processor count.
    \param count new processor count or 0 for default processor count.

    If set to 0, the processor count will be set to a suitable default on
    this target platform.

    If set to 1, processing will be forced to be single-threaded.

    The initial default is 0.
    */
   void setProcessorCount( int32 count );

   /** Returns the count of actual processors used by this VM.
       \return number of actual processors.

   In case the caller set the processor count to 0, a default count
   of processors is selected instead. That number is returned
   by this method (not the 0 setting).

   */
   int32 getProcessorCount() const;

   /** Starts and stops processors accordingly to current processor settings.
    */
   void updateProcessors();

   /** Gets a VM level message queue.
    \param Name the name of the message queue.

    If a message queue with the given name doesn't exist,
    it is created on the spot.

    Created message queues are never deleted (until the VM itself
    is deleted).
    */
   MessageQueue* getMessageQueue( const String& name );

   /** Gets the context manager associated with this virtual machine.  */
   ContextManager& contextManager() { return m_ctxMan; }
   const ContextManager& contextManager() const { return m_ctxMan; }

   const Scheduler& scheduler() const { return m_scheduler; }
   Scheduler& scheduler() { return m_scheduler; }


   //=========================================================
   // Utilities
   //=========================================================

   int32 getNextProcessID();
   int32 getNextContextID();
   int32 getNextGroupID();

   MTRand_interlocked& mtrand() const { return m_rand; }

   /** Get the VM-wide persistent data. */
   PData* pdata() const { return m_pdata; }

protected:

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;

   TextReader* m_textIn;
   TextWriter* m_textOut;
   TextWriter* m_textErr;
   Transcoder* m_stdCoder;

   ContextManager m_ctxMan;

   Scheduler m_scheduler;

   PData* m_pdata;

   mutable MTRand_interlocked m_rand;

private:
   void joinProcessors();

   int32 m_processorCount;
   class Private;
   VMachine::Private* _p;
};

}

#endif

/* end of vm.h */
