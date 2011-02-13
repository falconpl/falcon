/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.h

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_VMCONTEXT_H_
#define FALCON_VMCONTEXT_H_

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/codeframe.h>
#include <falcon/callframe.h>

#include <falcon/paranoid.h>

namespace Falcon {

class VMachine;

/**
 * Structure needed to store VM data.
 *
 * Note, VMContext is better not to have virtual members.
 *
 */
class FALCON_DYN_CLASS VMContext
{
public:
   static const int INITIAL_STACK_ALLOC = 256;
   static const int INCREMENT_STACK_ALLOC = 64;

   VMContext();
   ~VMContext();

   //=========================================================
   // Varaibles - stack management
   //=========================================================


   /** Return the nth variable in the local context.
    * Consider that:
    * - 0 is self.
    * - 1...N are the parameters
    * - N+1... are local variables.
    */
   const Item& localVar( int id ) const
   {
      return m_dataStack[ id + m_topCall->m_stackBase ];
   }

   Item& localVar( int id )
   {
      return m_dataStack[ id + m_topCall->m_stackBase ];
   }

   /** Return the nth parameter in the local context.
    *
    *TODO return 0 on parameter out of range.
    */
   inline const Item* param( int n ) const {
      return &m_dataStack[ n + m_topCall->m_stackBase ];
   }

   inline Item* param( int n )  {
      return const_cast<VMContext*>(this)->param( n );
   }

   /** Return the nth parameter in the local context.
    *
    *TODO use the local stack.
    */
   inline const Item* local( int n ) const {
      return &m_dataStack[ n + m_topCall->m_stackBase + m_topCall->m_paramCount ];
   }

   inline Item* local( int n ) {
      return &m_dataStack[ n + m_topCall->m_stackBase + m_topCall->m_paramCount ];
   }

   /** Push data on top of the stack */
   inline void pushData( const Item& data ) {
      ++m_topData;
      if( m_topData >= m_maxData )
      {
         moreData();
      }
      *m_topData = data;
   }

   /** Top data in the stack
    *
    */
   inline const Item& topData() const {
      return *m_topData;
   }

   inline Item& topData() {
      return *m_topData;
   }

   /** Removes the last element from the stack */
   void popData() {
      m_topData--;
      PARANOID( "Data stack underflow", (m_topData >= m_dataStack -1) );
   }

   inline long dataSize() const {
      return (m_topData - m_dataStack) + 1;
   }

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

   inline bool dataEmpty() const { return m_topData < m_dataStack; }

   void moreData();
   //=========================================================
   // Code frame management
   //=========================================================

   const CodeFrame& currentCode() const { return *m_topCode; }
   CodeFrame& currentCode() { return *m_topCode; }

   inline void popCode() {
      m_topCode--;
      PARANOID( "Code stack underflow", (m_topCode >= m_codeStack -1) );
   }

   void popCode( int size ) {
      m_topCode -= size;
      PARANOID( "Code stack underflow", (m_topCode >= m_codeStack -1) );
   }

   void unrollCode( int size ) {
      m_topCode = m_codeStack + size - 1;
      PARANOID( "Code stack underflow", (m_topCode >= m_codeStack -1) );
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

   /** Returns the current code stack size.
    *
    * During processing of SynTree and CompExpr, the change of the stack size
    * implies the request of a child item for the control to be returned to the VM.
    *
    */
   inline long codeDepth() const { return (m_topCode - m_codeStack) + 1; }

   /** Push some code to be run in the execution stack.
    *
    * The step parameter is owned by the caller.
    */
   inline void pushCode( const PStep* step ) {
      ++m_topCode;
      if( m_topCode >= m_maxCode )
      {
         moreCode();
      }
      m_topCode->m_step = step;
      m_topCode->m_seqId = 0;
   }

   void moreCode();

   //=========================================================
   // Stack resizing
   //=========================================================

   inline bool codeEmpty() const { return m_topCode < m_codeStack; }
   //=========================================================
   // Call frame management.
   //=========================================================

   /** Returns the self item in the local context.
    */
   inline const Item& self() const { return m_topCall->m_self; }
   inline Item& self() { return m_topCall->m_self; }

   const CallFrame& currentFrame() const { return *m_topCall; }
   CallFrame& currentFrame() { return *m_topCall; }

   const Item& regA() const { return m_regA; }
   Item& regA() { return m_regA; }

   inline long callDepth() const { return (m_topCall - m_callStack) + 1; }

   inline CallFrame* addCallFrame()  {
      ++m_topCall;
      if ( m_topCall >= m_maxCall )
      {
         moreCall();
      }
      return m_topCall;
   }

   void moreCall();

protected:

   // Inner constructor to create subclasses
   VMContext( bool );

   CodeFrame* m_codeStack;
   CodeFrame* m_topCode;
   CodeFrame* m_maxCode;

   CallFrame* m_callStack;
   CallFrame* m_topCall;
   CallFrame* m_maxCall;

   Item* m_dataStack;
   Item* m_topData;
   Item* m_maxData;

   Item m_regA;

   friend class VMachine;
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
