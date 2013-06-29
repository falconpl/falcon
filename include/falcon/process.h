/*
   FALCON - The Falcon Programming Language.
   FILE: process.h

   Falcon virtual machine -- process entity.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PROCESS_H_
#define _FALCON_PROCESS_H_

#include <falcon/setup.h>
#include <falcon/refcounter.h>
#include <falcon/mt.h>
#include <falcon/item.h>
#include <falcon/atomic.h>
#include <falcon/breakcallback.h>

namespace Falcon {

class VMachine;
class VMContext;
class Function;
class Closure;
class Stream;
class TextReader;
class TextWriter;
class Transcoder;
class Item;
class SynFunc;
class Error;
class GCLock;
class ModSpace;
class ItemPagePool;
class ItemStack;

class BreakCallback;

/** Process Entity.

 This entity represents an execution process in a virtual machine.

 The process has a VMContext which is the main context, it can be
 waited upon, interrupted asynchronously and checked for a result.

 The constructor is typically created by the virtual machine, which
 sets up a main context for the process and prepares it to run the
 desired code and/or symbol.

 The process can be used to either invoke the execution of a function
 (as i.e. the __main__ function of a module), or executable code stored
 in an arbitrary item, or it can be used to execute PSteps stored
 directly in the associated VM context.

 In the latter case, keep in mind that the Process class itself doesn't
 keep track of the PSteps; the calling program must ensure that the
 PStep instances are alive for the whole duration of the process.

 @note This can be easily ensured by subclassing the Process class
 and storing the required PSteps as process-specific data.
 */
class FALCON_DYN_CLASS Process
{
public:
   Process( VMachine* owner, ModSpace* hostSpace = 0);

   void adoptModSpace( ModSpace* hostSpace );

   int32 id() const { return m_id; }
   void id( int32 i ) { m_id = i; }

   VMachine* vm() const { return m_vm; }
   VMContext* mainContext() const { return m_context; }

   /** Opens the standard streams.
    *
    * This is a shotcut to putting the StdInStream() & c in the
    * standard streams known by this process.
    */
   void openStdStreams();

   /** Starts the process execution with the context configured as-is. */
   bool start();
   /** Starts the process invoking the given function. */
   bool start( Function* main, int pcount = 0, Item const* params = 0 );
   /** Starts the process invoking the given closure. */
   bool start( Closure* main, int pcount = 0, Item const* params = 0 );
   /** Starts the process invoking the given item. */
   bool startItem( Item& main, int pcount=0, Item const* params=0 );

   /**
    * Starts a context that is ready to run.
    *
    * The context should have been already created and configured,
    * eventually as a part of a context group, to call a function at
    * top of its code/call stack.
    */
   void startContext(VMContext* ctx);

   /** Called back by the context when going to be terminated.
    *
    */
   void onContextTerminated( VMContext* ctx );

   /** Returns the result of the evaluation.
    This is actually the topmost value in the stack of the main context.
    */
   const Item& result() const;
   Item& result();

   /**
    * Wait for the process to be complete.
    *
    * This call blocks until the timeout is expired, or until the process
    * is complete. If the process completes before the call is entered,
    * the method returns immediately.
    *
    * If the process is terminated with error, the error is
    * immediately thrown right after the wait is complete.
    *
    * After a successful wait, the process result can be obtained
    * by invoking the result() method.
    */
   InterruptibleEvent::wait_result_t wait( int32 timeout=-1 );

   /**
    * Interrupts a wait for the process completion.
    *
    * If a thread is currently held in wait for this process completion,
    * the wait function returns immediately with an appropriate "wait canceled"
    * return value.
    */
   void interrupt();

   /** This is called back by the main context when done.
    This will make wait non-blocking, or wakeup waits with success.
   */
   void completed();

   /** Declares that the process was terminated by an uncaught error.
    This will make wait non-blocking, or wakeup waits with success.
   */
   void completedWithError( Error* error );

   int32 getNextContextID();

   /**
    * Prepare the process with an entry point.
    * \return The entry point function.
    *
    * After calling this method, the process invoker can
    * put instructions to be executed in the main context,
    * and then invoke launch() as the code is complete.
    *
    * Alternatively, the caller of this method can synthesize
    * a syntactic tree in the returned syntactic function pointer,
    * but that will make this process to be usable just once.
    *
    * @note A void return PStep is automatically added at bottom of the
    * code stack to ensure a clean termination of the entry point
    * function.
    */
   SynFunc* readyEntry();

   /**
    * Returns the process termination error.
    * \return A valid error if the process was terminated with an error.
    *
    * In case the process is terminated with an error, this method will return
    * a valid error pointer.
    *
    * If this method returns 0 after the wait for process completion is complete,
    * then the process terminated correctly.
    */
   Error* error() const { return m_error; }

   void setResult( const Item& value );

   /** Gets the module space associated with this process. */
   ModSpace* modSpace() const { return m_modspace; }

   /** Terminates the process by asking termination of all the contexts.
    *
    * This will cause the process to exit as soon as possible. External
    * entities waiting on the process termination will not be waken up
    * until the process really exits the virtual machine, unless
    * interrupt() is also called.
    */
   void terminate();

   /** Terminates the process by asking termination of all the contexts.
    *
    * This will cause the process to exit as soon as possible. External
    * entities waiting on the process termination will not be waken up
    * until the process really exits the virtual machine, unless
    * interrupt() is also called.
    */
   void terminateWithError( Error* error );

   /** Returns true if this process has been terminated.

      This is used by the virtual machine to interrupt
      any operation on any context belonging to this process.
    */

   bool isTerminated() const { return atomicFetch(m_terminated) > 0; }

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
     \see bool setStdEncoding( const String& name )
     */
   void setStdEncoding( Transcoder* ts );

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
   inline TextWriter* textErr() const { return m_textOut; }

   /**
    * Used internally
    */
   void gcMark( uint32 mark ) { m_mark = mark; }
   /**
    * Used internally
    */
   uint32 currentMark() const { return m_mark; }


   /** Changes the string table for i-string translation.
    * \param dict A string->string dictionary of translations.
    * \return true if the dictionary has all strings, false if some item is not a string.
    *
    * This method receives an item dictionary that should contain
    * strings only. If not, the method returns false.
    */
   bool setTranslationsTable( ItemDict* dict, bool bAdditive = false );

   /**
    * Adds a new translation string.
    *
    * Translations are not immediately visible; it's necessary to invoke
    * commitTranslation() to make the added translations to be seen
    * in the global translation table.
    */
   void addTranslation( const String& original, const String& tld );

   void commitTranslations( bool additive = false );

   class TranslationEnumerator
   {
   public:
      TranslationEnumerator() {}
      virtual ~TranslationEnumerator() {}
      virtual bool count( size_t size ) = 0;
      virtual void operator()( const String& orig, const String& tl ) = 0;
   };

   /** Enumerates all the translations in the current table.
    * The enumeration loop is performed outside the translation lock,
    * after having done a copy of the translation table.
    *
    * This method will first invoke the count() method of the translation enumerator
    * to notify the size of the table. If that method returns false, the
    * enumeration is not performed.
    *
    */
   void enumerateTranslations( TranslationEnumerator &te );

   /** Gets the translation of a given string.
    * \param code The untraslated original string.
    * \param tld A string where to place the translation.
    * \param bool True if the translation is found, false otherwise.
    *
    * If the translation is not found, tld is unmodified.
    */
   bool getTranslation( const String& original, String& tld ) const;

   /** Gets the translation of a given string for syntactic rendering.
    * \param code The untraslated original string.
    * \param tld A string where to place the translation.
    * \param gen The generation known by the invoker (0 for unknown).
    * \param bool True if the generation is changed, false otherwise.
    *
    * This version of the function checks if the translation of a given original
    * string has changed since last time the invoker has asked it.
    *
    * When the invoker hasn't any previous result for this function, it invokes
    * the function with gen=0.
    *
    * When the generation is different from the one currently known
    * by the engine, this method always fills tld.
    * If a translation for the original string is not
    * found, tld receives a copy of the string passed as original. In any case,
    * when \b gen is different from the current generation, the method returns
    * true, indicating that the caller should update its cache. Also, \b gen
    * is changed to the currently known generation.
    *
    * When the \b gen is the same as the generation known by this process,
    * the string in tld is unchanged and the method returns false.
    *
    * \note The initial status of the process sets the current generation to 1;
    * this means that the first time the method is invoked, all the i-strings of
    * a process will be updated (with copies of their original). The first change
    * of the string table for the process is marked with generation 2.
    *
    */
   bool getTranslation( const String& original, String& tld, uint32 &gen) const;

   /** Gets the current translation generation.
    */
   uint32 getTranslationGeneration() const;

   /** Pool for item pages */
   Pool* itemPagePool() const { return m_itemPagePool; }

   /** Adds an export to the process superglobals table.
    * \param name The name of the exported symbol.
    * \param value The value at which the exported symbol must be initialized.
    * \return the pointer to the exported item if the export was added,
    *       0 if the exported symbol name was already defined.
    *
    * The exported item is also garbage-locked, keeping the inner value
    * safe as long as the process exists.
    *
    * The item itself is stored in a non-relocable table which is actually
    * never deleted.
    *
    * Once an export is declared, it cannot be re-declared; however, the
    * value in the exported item can be changed at will.
    */
   Item* addExport( const String& name, const Item& value );

   /** Removes an export from the process superglobals table.
    * \param name The name of the exported symbol to be removed.
    * \return true if the name was found and removed, false otherwise.
    */
   bool removeExport( const String& name );

   /** Retrieves the value of an exported name.
    * \param name The name of the exported symbol to be found in the superglobals.
    * \return a valid Item* on success, 0 otherwise.
    */
   Item* getExport( const String& name ) const;

   /** Atomically alters an exported value, or create it if not previously exported.
    * \param name The name of the exported symbol to be found in the superglobals.
    * \param value The new value for the item.
    * \param existing will be set to true if the item was already in the exported table.
    * \return a valid Item* to the exported value.
    */
   Item* updateExport( const String& name, const Item& value, bool &existing ) const;

   /** Sets (or removes) a break callback handler.
    * \param bcb The new break callback, or zero to remove previous callbacks.
    *
    * The new break callback receives a BreakCallback::onInstalled message, while
    * the old one receives a onUninstalled message.
    *
    * \note It's responsibility of the host
    * code to ensure that the BreakCallback entity stays alive long enough. Also,
    * there is no protection against concurrency, so the onUninstalled message could
    * be generated even while the BreakCallback is currently being consulted; if necessary,
    * use an internal reference counter to deal with the race condition.
    *
    */
   void setBreakCallback( BreakCallback* bcb );

   /** Invoked when a processor finds a breakpoint.
    * \return True if a break callback handler was set and invoked, false otherwise.
    *
    * It is granted that the method will return coherently and atomically considering
    * the currently installed break callback routine.
    */
   bool onBreakpoint( Processor* prc, VMContext* ctx );

protected:
   Process( VMachine* owner, bool added );
   virtual ~Process();

   void launch();
   bool checkRunning();

   friend class VMachine;
   int32 m_id;
   VMachine* m_vm;
   VMContext *m_context;
   InterruptibleEvent m_event;
   SynFunc* m_entry;
   atomic_int m_terminated;

   bool m_running;
   mutable Mutex m_mtxRunning;
   mutable Mutex m_mtxContexts;

   atomic_int m_ctxId;
   Error* m_error;
   bool m_added;

   Item m_result;
   GCLock* m_resultLock;

   ModSpace* m_modspace;

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;

   TextReader* m_textIn;
   TextWriter* m_textOut;
   TextWriter* m_textErr;

   Transcoder* m_stdCoder;
   uint32 m_mark;

   uint32 m_tlgen;

   Pool* m_itemPagePool;
   ItemStack* m_superglobals;

   BreakCallback* m_breakCallback;
   Mutex m_mtxBcb;

   void inheritStreams();

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Process)

   class Private;
   Private* _p;
};

}
#endif

/* end of process.h */
