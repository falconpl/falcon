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
    */
   VMachine( Stream* stdIn = 0, Stream* stdOut=0, Stream* stdErr = 0 );
   virtual ~VMachine();

   //=========================================================
   // Context management
   //=========================================================

   inline VMContext* currentContext() const { return m_context; }

   //=========================================================
   // Execution management.
   //=========================================================

   /** Runs a prepared code, or continues running.
    * @return True if all the code was executed, false if a breakpoint forced to stop.
    *
    * This method runs the virtual machine until:
    * - All the code in the code stack is executed; or
    * - a break flag is set.
    *
    * In the second case, the method returns false.
    *
    * Notice that the effect of exit() is that of terminating the VM
    * cleanly; this will make this function to return true.
    */
   bool run();


   //=========================================================
   // Debug support
   //=========================================================

   /** Returns the step that is going to be executed next, or null if none.
    \return The next step that will be executed.
    */
   const PStep* nextStep() const;

   /** Performs a single step.
      @return true if there is another step ready to be executed,
         false if this was the last (i.e. if the VM is terminated).
    */
   bool step();

      /** Gives a description of the location of the next step being executed.
    @return A string with a textual description of the source position of the
            next step being executed.

    This is a debug function used to indicate where is the next step being
    executed in a source file.
   */
   String location() const;

   /** Outlines VM status in a string.
    @return A string with a textual description of the VM status.

    This is a debug function used to indicate what's the current status of the
    virtual machine.
   */
   String report();

   /** Gives a description of the location of the next step being executed.
    @param infos An instance of LocationInfo receiving the debug information about
           the location in the source files of the next step.

    This information is more detailed and GUI oriented than the information
    returned by location().
    */
   bool location( LocationInfo& infos ) const;


   //=========================================================
   // General information.
   //=========================================================

   /** Raises a VM error.
    *
    * The method finds a try frame back in the code stack, and if it is found,
    * the control is moved to a suitable catch item.
    *
    * If a suitable catch frame is not found, the error is thrown as a C++ exception.
    *
    * \note Falcon exceptions are thrown by pointer. This is because the error entity
    * can be stored in several places, and in several threads, by the time it
    * surfaces to user code. This calls for reference management.
    */
   //void raiseError( Error* theError );


   /** Finds the variable corresponding to a symbol name in the current context.
    * @return A pointer to the item if found, 0 if not found.
    *
    * The search will be extended to global and imported symbols if the search
    * in the local symbol tables fails.
    */
   Item* findLocalItem( const String& name ) const;

   /** Returns true if the current has not any code.

    */
   inline bool codeEmpty() const { return m_context->codeEmpty(); }

   /** Sets a value in the A register of the current context.
    \param v The return value.
    \deprecated  Kept for compatibility with engine 0.9.x

    This method sets a value in the A register of the current context.
    In the old vm 0.9.x this worked as a return value, and it is still
    interpreted this way in the implementation of the ExtFunction 
    class (which wraps the old functions).

    New code should use VMContext::returnFrame().
    */
   void retval( const Item& v ) { currentContext()->regA() = v; }


   /** Access the current context accumulator. */
   const Item& regA() const { return m_context->regA(); }
   /** Access the current context accumulator. */
   Item& regA() { return m_context->regA(); }

   /** Access the current "self" item. */
   const Item& self() const { return m_context->self(); }
   /** Access the current context "self" item. */
   Item& self() { return m_context->self(); }


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
   void setStdEncoding( Transcoder* ts, bool bOwn = false );

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

   //=========================================================
   // Module management
   //=========================================================

   /** Gets the module space associated with this virtual machine. */
   ModSpace* modSpace() const { return m_modspace; }

protected:

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;

   TextReader* m_textIn;
   TextWriter* m_textOut;
   TextWriter* m_textErr;

   Transcoder* m_stdCoder;
   bool m_bOwnCoder;
   
   ModSpace* m_modspace;

   /** Called back when an error was thrown directly inside the machine.
    \param e The error being thrown.
    */
   void onError( Error* e );

   /** Called back after the main loop gets a raiseItem().
    \param item The item being raised.
    */
   void onRaise( const Item& item );
   
private:
   // current context
   VMContext* m_context;
   
   
   
   class Private;
   Private* _p;   
};

}

#endif

/* end of vm.h */
