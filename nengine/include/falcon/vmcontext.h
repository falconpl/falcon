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

   VMContext( VMachine* owner = 0 );
   ~VMContext();

   void assign( VMachine* vm )
   {
      m_vm = vm;
   }

   VMachine* vm() const { return m_vm; }
   
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


   inline Item* opcodeParams( int count )
   {
      return m_topData - (count-1);
   }

   inline Item& opcodeParam( int count )
   {
      return *(m_topData - count);
   }
   
   inline const Item& opcodeParam( int count ) const
   {
      return *(m_topData - count);
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

   /** Add more variables on top of the stack.
    \param count Number of variables to be added.
    The resulting values will be nilled.
    */
   inline void addLocals( size_t count ) {
      Item* base = m_topData+1;
      m_topData += count;
      if( m_topData >= m_maxData )
      {
         moreData();
         base = m_topData - count;
      }
      while( base <= m_topData )
      {
         base->setNil();
         ++base;
      }
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
      if( m_topData >= m_maxData )
      {
         moreData();
      }
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

         vals = param(-1)->asInteger();
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
      topCall->m_bInit = false;

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

//========================================================
//


   /** Gets the operands from the top of the stack.
    \param op1 The first operand.
    \see Class

    \note this method may be used also by pseudofunctions and generically
    by any PStep in need to access the top of the stack.
    */
   inline void operands( Item*& op1 )
   {
      op1 = &topData();
   }

   /** Gets the operands from the top of the stack.
    \param op1 The first operand.
    \param op2 The second operand.
    \see Class

    \note this method may be used also by pseudofunctions and generically
    by any PStep in need to access the top of the stack.
    */
   inline void operands( Item*& op1, Item*& op2 )
   {
      op1 = &topData()-1;
      op2 = op1+1;
   }

   /** Gets the operands from the top of the stack.
    \param op1 The first operand.
    \param op2 The second operand.
    \param op3 The thrid operand.
    \see Class

    \note this method may be used also by pseudofunctions and generically
    by any PStep in need to access the top of the stack.
    */
   inline void operands( Item*& op1, Item*& op2, Item*& op3 )
   {
      op1 = &topData()-2;
      op2 = op1+1;
      op3 = op2+1;
   }

   /** Pops the stack when leaving a PStep or operand, and sets an operation result.
    \param count Number of operands accepted by this step
    \param result The value of the operation result.
    \see Class

    \note this method may be used also by pseudofunctions and generically
    by any PStep in need to access the top of the stack.
    */
   inline void stackResult( int count, const Item& result )
   {
      if( count > 1 ) popData( count-1 );
      topData() = result;
   }

      /** Returns the parameter array in the current frame.
    \return An array of items pointing to the top of the local frame data stack.

    This method returns the values in the current topmost data frame.
    This usually points to the first parameter of the currently executed function.
    */
   inline Item* params() const { return params(); }

   /** Returns pseudo-parameters.
    \param count Number of pseudo parameters.
    \return An array of items pointing to the pseudo parameters.

    This can be used to retrieve the parameters of pseudo functions.
    */
   inline Item* pseudoParams( int32 count ) {
      return &topData() - count + 1;
   }

   //=========================================================
   // Deep call protocol
   //=========================================================

   /** Deep calls operand protocol -- part 1.

    The deep call protocol is used by falcon Function instances or other
    PStep* operands that are called by the virtual machine, and that may then
    call the virtual machine again.

    Some functions called by the virtual machine then need to call other functions that
    may or may not need the VM to perform other calculations.

    If a calculation that involves the Virtual Machine terminates immediately,
    the result is usually found in VMachine::topData(), and the caller can
    progress immediately.

    Otherwise, the caller needs to set a callback in the virtual machine so that
    it will be called back by it after the frame added by the called is
    entity is complete. Then, the result will be available in regA().

    But, the step should be on top of the code stack before the underluing called
    element has a chance to prepare its call frame. This means that normally a
    caller calling an operand that may or may not request a call frame should
    push its own callback on the code stack blindly, then eventually pop it if
    the called entity didn't push a new call frame.

    To avoid this on the most common situations where this may be required, a
    set of three metods are used by the operand implementations and a set of
    well defined functions known by the engine in the virtual machine.

    The caller sets a (non-destructible) PStep through the ifDeep() method.
    If the callee wants to add a frame, it calls goingDeep(), which does nothing
    if the caller didn't need to have a result from the called, but will store the
    PStep in the code stack otherwise.

    Then, the caller must check if the result is ready in topData() or if the
    processing must be delayed by calling wentDeep(); that method will also clear
    the readied PStep.

    To avoid this mechanism to be broken, ifDeep() raises an error if called
    while another PStep was readied.

    Of course, this mechanism cannot be used across context switches, but its
    meant for tightly coupled functions.
    */
   void ifDeep( const PStep* postcall );

   /** Called by a callee in a possibly deep call pair.
    @see ifDeep.
   */
   void goingDeep();

   /** Called by a calling method to know if the called sub-methdod required a deep operation.
    \return true if the sub-method needed to go deep.
    */
   bool wentDeep();


   /** Pushes a quit step at current position in the code stack.
    This method should always be pushed in a VM before it is invoked
    from unsafe code.
    */
   void pushQuit();

   /** Pushes a breakpoint at current postition in the code stack.
    */
   void pushBreak();

   /** Pushes a VM clear suspension request in the code stack.
    @see setReturn
    */
   void pushReturn();


   void call( Function* f, int np, bool bExpr = false );

   /** Prepares the VM to execute a function (actually, a method).

    The VM gets readied to execute the function from the next step,
    which may be invoked via step(), run() or by returning from the caller
    of this function in case the caller has been invoked by the VM itself.
    @param function The function to be invoked.
    @param np Number of parameters that must be already in the stack.
    @param self The item on which this method is invoked. Pure functions are
                considered methods of "nil".
    */
   virtual void call( Function* function, int np, const Item& self, bool bExpr = false );

   /** Returns from the current frame */
   void returnFrame();

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

   VMachine* m_vm;

   friend class VMachine;
   friend class SynFunc;

   // used by ifDeep - goingDeep() - wentDeep() triplet
   const PStep* m_deepStep;
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
