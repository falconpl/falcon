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
class SynFunc;

/**
 * Structure needed to store VM data.
 *
 * Note, VMContext is better not to have virtual members.
 *
 */
class FALCON_DYN_CLASS VMContext
{
public:
   static const int INITIAL_STACK_ALLOC = 512;
   static const int INCREMENT_STACK_ALLOC = 256;

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
    */
   inline const Item* param( int n ) const {
      fassert(m_dataStack+(n + m_topCall->m_stackBase) < m_maxData );
      if( m_topCall->m_paramCount <= n ) return 0;
      return &m_dataStack[ n + m_topCall->m_stackBase ];
   }

   inline Item* param( int n )  {
      fassert(m_dataStack+(n + m_topCall->m_stackBase) < m_maxData );
      if( m_topCall->m_paramCount <= n ) return 0;
      return &m_dataStack[ n + m_topCall->m_stackBase ];
   }

   inline Item* params() {
      return &m_dataStack[ m_topCall->m_stackBase ];
   }

   /** Return the nth parameter in the local context.
    *
    *TODO use the local stack.
    */
   inline const Item* local( int n ) const {
      fassert(m_dataStack+(n + m_topCall->m_stackBase + m_topCall->m_paramCount) < m_maxData );
      return &m_dataStack[ n + m_topCall->m_stackBase + m_topCall->m_paramCount ];
   }

   inline Item* local( int n ) {
	  fassert(m_dataStack+(n + m_topCall->m_stackBase + m_topCall->m_paramCount) < m_maxData );
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
      fassert( m_topData >= m_dataStack && m_topData < m_maxData );
      return *m_topData;
   }

   inline Item& topData() {
       fassert( m_topData >= m_dataStack && m_topData < m_maxData);
       return *m_topData;
   }

   /** Removes the last element from the stack */
   void popData() {
      m_topData--;
      PARANOID( "Data stack underflow", (m_topData >= m_dataStack -1) );
   }

   /** Removes the last n elements from the stack */
   void popData( size_t size ) {
      m_topData-= size;
      PARANOID( "Data stack underflow", (m_topData >= m_dataStack -1) );
   }

   inline long dataSize() const {
      return (m_topData - m_dataStack) + 1;
   }

   inline long localVarCount() const {
      return ((m_topData+1) - m_dataStack) - currentFrame().m_stackBase;
   }

   /** Copy multiple values in a target. */
   void copyData( Item* target, size_t count, size_t start = (size_t)-1);

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

   /** Enlarge the data stack.*/
   void moreData();


   /** Begins a new rule frame.
    
    This is called when a new rule is encountered to begin the rule framing.
    */
   void startRuleFrame();

   /** Adds a new local rule frame with a non-deterministic traceback point.
    \param tbPoint a traceback point or 0xFFFFFFFF if there isn't any
            traceback point (topmost rule frame in a rule).

    This creates a copy of the parameters and local variables of the
    current call context, and shifts the stackbase forward.
    */
   void addRuleNDFrame( uint32 tbPoint );

   /** Retract the current rule status and resets the rule frame to its previous state.
    */
   inline uint32 unrollRuleFrame()
   {
      CallFrame& callf = currentFrame();

      // assert if we're not in a rule frame.
      fassert( callf.m_stackBase > callf.m_initBase );

      // our frame informations are at param -1
      int64 vals = param(-1)->asInteger();

      // roll back to previous state stack state.
      m_topData = m_dataStack + callf.m_stackBase - 2;
      callf.m_stackBase = (int32) (vals >> 32);

      // return the traceback part
      return (uint32)(vals & 0xFFFFFFFF);
   }

   /** Retract all the ND branches and get back
    */
   inline void unrollRuleBranches()
   {
      CallFrame& callf = currentFrame();

      // assert if we're not in a rule frame.
      fassert( callf.m_stackBase > callf.m_initBase );

      // our frame informations are at param -1
      int64 vals = param(-1)->asInteger();
      while( (vals & 0xFFFFFFFF) != 0xFFFFFF )
      {
         m_topData = m_dataStack + callf.m_stackBase - 2;

         // roll back to previous state stack state.
         callf.m_stackBase = (int32) (vals >> 32);

         int64 vals = param(-1)->asInteger();
         // assert if we're not in a rule frame anymore!
         fassert( callf.m_stackBase > callf.m_initBase );
      }
   }

   /** Retract a whole rule, thus closing it.
    */
   inline void unrollRule()
   {
      CallFrame& cf = currentFrame();
      // assert if we're not in a rule frame.
      fassert( cf.m_stackBase > cf.m_initBase );

      long localCount = localVarCount();
      int32 baseRuleTop = param(-1)->content.mth.ruleTop;
      // move forward the stack base.
      cf.m_stackBase = baseRuleTop;
      m_topData = m_dataStack + baseRuleTop + localCount - 1;

   }

   /** Commits a whole rule, thus closing it.
    */
   void commitRule();

   /** Specicfy that the current context is ND. 
    When inside a rule, activating this bit will cause the upper rule to
    consider the current operation as non-deterministic, and will cause a new
    non-deterministic frame to be generated as the statement invoking this
    method returns.

    The method has no effect when not called from a non-rule context
    */
   void SetNDContext()
   {
      register CallFrame& cf = currentFrame();
      // are we in a rule frame?
      if( cf.m_initBase < cf.m_stackBase )
      {
         param(-1)->setOob();
      }
   }

   /** Checks (and clear) non-deterministic contexts.

    A rule will check if a statement has performed some non-deterministic operation
    calling this method, that will also clear the determinism status so no further
    operation needs to be called by the checking rule.

    */
   bool checkNDContext()
   {
      register CallFrame& cf = currentFrame();
      if( cf.m_initBase < cf.m_stackBase )
      {
         register Item& itm = *param(-1);
         register bool mode = itm.isOob();
         itm.resetOob();
         return mode;
      }
      return false;
   }

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

   /** Sets a value as return value for the current function.
    */
   void retval( const Item& v ) { m_regA = v; }

   inline long callDepth() const { return (m_topCall - m_callStack) + 1; }

   inline CallFrame* addCallFrame()  {
      ++m_topCall;
      if ( m_topCall >= m_maxCall )
      {
         moreCall();
      }
      return m_topCall;
   }

   /** Prepares a new methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams, const Item& self, bool isExpr )
   {
      register CallFrame* topCall = addCallFrame();
      topCall->m_function = function;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      topCall->m_paramCount = nparams;
      topCall->m_self = self;
      topCall->m_bMethodic = true;
      topCall->m_bExpression = isExpr;

      return topCall;
   }

   /** Prepares a new non-methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams, bool isExpr )
   {
      register CallFrame* topCall = addCallFrame();
      topCall->m_function = function;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      topCall->m_paramCount = nparams;
      topCall->m_self.setNil();
      topCall->m_bMethodic = false;
      topCall->m_bExpression = isExpr;

      return topCall;
   }

   bool isMethodic() const { return m_topCall->m_bMethodic; }
   
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
   friend class SynFunc;

   // used by ifDeep - goingDeep() - wentDeep() triplet
   const PStep* m_deepStep;
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
