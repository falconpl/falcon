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

/** The Falcon virtual machine.
*/
class FALCON_DYN_CLASS VMachine
{

public:
   VMachine();
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

   inline void call( Function* f, int np )
   {
      call( f, np, Item() );
   }

   virtual void call( Function* function, int np, const Item& self );

   void returnFrame();

   void report( String &data );

   /** Returns the step that is going to be executed next, or null if none */
   PStep* nextStep() const;

   /** Performs a single step.
    * @return true if there is another step ready to be executed,
    * false if this was the last (i.e. if the VM is terminated).
    */
   bool step();


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
   Item* findLocalItem( const String& name );

   /** Returns true if the current has not any code.

    */
   inline bool codeEmpty() const { return m_context->codeEmpty(); }

   const Item& regA() const { return m_context->regA(); }
   Item& regA() { return m_context->regA(); }
protected:

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;
   bool m_bhasStandardStreams;

   void internal_construct();

private:

   // current context
   VMContext* m_context;

   // True when an event is set.
   enum {
      eventNone,
      eventBreak,
      eventTerminate
   };
};

}

#endif

/* end of vm.h */
