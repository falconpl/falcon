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

#include <vector>

#define FALCON_VM_DFAULT_CHECK_LOOPS 5000

namespace Falcon {

/** The Falcon virtual machine.
*/
class FALCON_DYN_CLASS VMachine: public BaseAlloc
{

public:
   VMachine();
   virtual ~VMachine();

   /** Return the nth variable in the local context.
    * Consider that:
    * - 0 is self.
    * - 1...N are the parameters
    * - N+1... are local variables.
    */
   const Item& localVar( int id ) const
   {
      const CallFrame& cs = m_callStack.back();
      return m_dataStack[ id + cs.m_stackBase ];
   }

   Item& localVar( int id )
   {
      const CallFrame& cs = m_callStack.back();
      return m_dataStack[ id + cs.m_stackBase ];
   }

   /** Return the nth parameter in the local context.
    *
    *TODO return 0 on parameter out of range.
    */
   inline const Item* param( int n ) const {
      const CallFrame& cs = m_callStack.back();
      return &m_dataStack[ n + cs.m_stackBase ];
   }

   inline  Item* param( int n )  {
      const CallFrame& cs = m_callStack.back();
      return &m_dataStack[ n + cs.m_stackBase ];
   }

   /** Return the nth parameter in the local context.
    *
    *TODO use the local stack.
    */
   inline const Item* local( int n ) const {
      const CallFrame& cs = m_callStack.back();
      return &m_dataStack[ n + cs.m_paramCount + cs.m_stackBase ];
   }

   inline Item* local( int n ) {
         const CallFrame& cs = m_callStack.back();
         return &m_dataStack[ n + cs.m_paramCount + cs.m_stackBase ];
      }

   /** Returns the self item in the local context.
    *
    */
   inline const Item& self() const { return m_callStack.back().m_self; }
   inline Item& self() { return m_callStack.back().m_self; }

   /** Top data in the stack
    *
    */
   inline const Item& topData() const { return *m_topData; }
   inline Item& topData() { return *m_topData; }

   /** Push some code to be run in the execution stack.
    *
    * The step parameter is owned by the caller.
    */
   inline void pushCode( const PStep* step ) {
      ++m_topCode;
      m_topCode->m_step = step;
      m_topCode->m_seqId = 0;
   }

   /** Push data on top of the stack */
   inline void pushData( const Item& data ) {
      ++m_topData;
      *m_topData = data;
   }

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
    * surfaces to user code. This calls for referecne management.
    */
   //void raiseError( Error* theError );

   const CallFrame& currentFrame() const { return m_callStack[m_callStack.size()-1]; }
   CallFrame& currentFrame() { return m_callStack[m_callStack.size()-1]; }

   const CodeFrame& currentCode() const { return *m_topCode; }
   CodeFrame& currentCode() { return *m_topCode; }

   void popData() { m_topData--; }

   inline void popCode() {
      popCode(1);
   }

   /** Changes the currently running pstep.
    *
    *  Other than changing the top step, this method resets the sequence ID.
    *  Be careful: this won't work for PCode (they need the seq ID to be set to their size).
    */
   inline void resetCode( PStep* step ) {
      CodeFrame& frame = currentCode();
      frame.m_step = step;
      frame.m_seqId = 0;
   }

   void popCode( int size ) {
      m_topCode -= size;
   }

   void unrollCode( int size ) {
      m_topCode = m_codeStack + size - 1;
   }

   void returnFrame();

   /** Adds a data in the stack and returns a reference to it.
    *
    * Useful in case the caller is going to push some data in the stack,
    * but it is still not having the final item ready.
    *
    * Using this method and then changing the returned item is more
    * efficient than creating an item in the caller stack
    * and then pushing it in the VM.
    *
    */
   Item& addDataSlot() {
      ++m_topData;
      return *m_topData;
   }

   /** Finds the variable coresponding to a symbol name in the current context.
    * @return A pointer to the item if found, 0 if not found.
    *
    * The search will be extended to global and imported symbols if the search
    * int he local symbol tables fails.
    */
   Item* findLocalItem( const String& name );

   /** Returns the current code stack size.
    *
    * During processing of SynTree and CompExpr, the change of the stack size
    * implies the request of a child item for the control to be returned to the VM.
    *
    */
   int codeDepth() const { return (m_topCode - m_codeStack) + 1; }

   int dataSize() const { return (m_topData - m_dataStack) + 1; }
protected:

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;
   bool m_bhasStandardStreams;

   void internal_construct();

private:

   //typedef std::vector<CodeFrame> CodeStack;
   typedef std::vector<CallFrame> CallStack;
   //typedef std::vector<Item> DataStack;

   //CodeStack m_codeStack;
   CodeFrame* m_codeStack;
   CodeFrame* m_topCode;

   CallStack m_callStack;
   //DataStack m_dataStack;
   Item* m_dataStack;
   Item* m_topData;

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
