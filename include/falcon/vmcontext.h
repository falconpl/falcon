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
class StmtTry;
class SynTree;
class DynSymbol;

/**
 * Structure needed to store VM data.
 *
 * Note, VMContext is better not to have virtual members.
 *
 */
class FALCON_DYN_CLASS VMContext
{
public:
   VMContext( VMachine* owner = 0 );
   ~VMContext();

   void assign( VMachine* vm )
   {
      m_vm = vm;
   }

   VMachine* vm() const { return m_vm; }
   
   /** Resets the context to the initial state.
    
    This clears the context and sets it as if it was just created.
    */
   void reset();
   
   /** Sets the topmost code as "safe".
    In interactive mode, when what's done up to date is correct,
    an error must not cause the invalidation of the whole code structure.
    
    For instance,
    @code
    x = 1
    if x
      x = 2
      an error  // raises a grammar error
    end
    > x        // expected to be 2
    
    After each "stop point", usually a self contained statement, the
    interactive compiler calls this method to mark a point where everything
    was considered ok. In case of subsequent error, the code stack (and
    eventually other stacks across function borders) is rolled back to this
    point.
    */
   void setSafeCode();
   
   //=========================================================
   // Varaibles - stack management
   //=========================================================


   /** Return the nth variable in the local context.
    * Consider that:
    * - 0...N-1 are the parameters
    * - N... are local variables.
    */
   const Item& localVar( int id ) const
   {
      return m_dataStack.m_base[ id + currentFrame().m_stackBase ];
   }

   Item& localVar( int id )
   {
      return m_dataStack.m_base[ id + currentFrame().m_stackBase ];
   }

   /** Return the nth parameter in the local context.
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline const Item* param( uint32 n ) const {
      fassert(m_dataStack.m_base+(n + currentFrame().m_stackBase) < m_dataStack.m_max );
      if( currentFrame().m_paramCount <= n ) return 0;
      return &m_dataStack.m_base[ n + currentFrame().m_stackBase ];
   }

   /** Return the nth parameter in the local context (non-const).
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline Item* param( uint32 n )  {
      fassert(m_dataStack.m_base+(n + currentFrame().m_stackBase) < m_dataStack.m_max );
      if( currentFrame().m_paramCount <= n ) return 0;
      return &m_dataStack.m_base[ n + currentFrame().m_stackBase ];
   }

  /** Returns the parameter array in the current frame.
    \return An array of items pointing to the top of the local frame data stack.

    This method returns the values in the current topmost data frame.
    This usually points to the first parameter of the currently executed function.
    */
   inline Item* params() {
      return &m_dataStack.m_base[ currentFrame().m_stackBase ];
   }

   inline int paramCount() {
      fassert( &currentFrame() >= m_callStack.m_base );
      return currentFrame().m_paramCount;
   }


   inline Item* opcodeParams( int count )
   {
      return m_dataStack.m_top - (count-1);
   }

   inline Item& opcodeParam( int count )
   {
      return *(m_dataStack.m_top - count);
   }
   
   inline const Item& opcodeParam( int count ) const
   {
      return *(m_dataStack.m_top - count);
   }

   /** Return the nth parameter in the local context.
    *
    *TODO use the local stack.
    */
   inline const Item* local( int n ) const {
      fassert(m_dataStack.m_base+(n + currentFrame().m_stackBase + currentFrame().m_paramCount) < m_dataStack.m_max );
      return &m_dataStack.m_base[ n + currentFrame().m_stackBase + currentFrame().m_paramCount ];
   }

   inline Item* local( int n ) {
	  fassert(m_dataStack.m_base+(n + currentFrame().m_stackBase + currentFrame().m_paramCount) < m_dataStack.m_max );
      return &m_dataStack.m_base[ n + currentFrame().m_stackBase + currentFrame().m_paramCount ];
   }

   /** Push data on top of the stack.
    \item data The data that must be pushed in the stack.
    \note The data pushed is copied by-value in a new stack element, but it is
          not colored.
    */
   inline void pushData( const Item& data ) {
      ++m_dataStack.m_top;
      if( m_dataStack.m_top >= m_dataStack.m_max )
      {
         Item temp = data;
         m_dataStack.more();
         *m_dataStack.m_top = temp;
      }
      else {
         *m_dataStack.m_top = data;
      }
   }

   /** Add more variables on top of the stack.
    \param count Number of variables to be added.
    The resulting values will be nilled.
    */
   inline void addLocals( size_t count ) {
      Item* base = m_dataStack.m_top+1;
      m_dataStack.m_top += count;
      if( m_dataStack.m_top >= m_dataStack.m_max )
      {
         m_dataStack.more();
         base = m_dataStack.m_top - count;
      }
      while( base <= m_dataStack.m_top )
      {
         base->setNil();
         ++base;
      }
   }

   /** Add more variables on top of the stack -- without initializing them to nil.
    \param count Number of variables to be added.

    This is like addLocals, but doesn't nil the newly created variables.
    */
   inline void addSpace( size_t count ) {
      m_dataStack.m_top += count;
      if( m_dataStack.m_top >= m_dataStack.m_max )
      {
         m_dataStack.more();
      }
   }
   
   /** Insert some data at some point in the stack.
    \param pos The position in the stack where data must be inserted, 0 being top.
    \param data The data to be inserted.
    \param dataSize Count of items to be inserted.
    \param replSize How many items should be overwritten starting from pos.
    
    Suppose you want to shift some parameters to make room for a funciton, and
    eventually extra parameters, before the function call. For instance, suppose
    you have...
    
    \code
    ..., (data), (data), (param2), (param3) // top
    \endcode
    
    And want to add the function and an extra parameter:
    
    \code
    ..., (data), (data), [func], [param1], (param2), (param3) // top
    \endcode
    
    This mehtod allows to shift forward the nth item in the stack and 
    place new data. For instance, the code to perform the example operation
    would be:
    \code
    {
       ...
       Item items[] = { funcItem, param1 };
       ctx->insertData( 2, items, 2 );
       ...
    }
    \endcode    
    */
   void insertData( int32 pos, Item* data, int dataSize, int32 replSize=0 );

   /** Top data in the stack
    *
    */
   inline const Item& topData() const {
      fassert( m_dataStack.m_top >= m_dataStack.m_base && m_dataStack.m_top < m_dataStack.m_max );
      return *m_dataStack.m_top;
   }

   inline Item& topData() {
       fassert( m_dataStack.m_top >= m_dataStack.m_base && m_dataStack.m_top < m_dataStack.m_max);
       return *m_dataStack.m_top;
   }

   /** Removes the last element from the stack */
   inline void popData() {
      m_dataStack.pop();
      // Notice: an empty data stack is an error.
      PARANOID( "Data stack underflow", (m_dataStack.m_top >= m_dataStack.m_base) );
   }

   /** Removes the last n elements from the stack */
   inline void popData( int size ) {
      m_dataStack.pop(size);
      // Notice: an empty data stack is an error.
      PARANOID( "Data stack underflow", (m_dataStack.m_top >= m_dataStack.m_base) );
   }

   inline long dataSize() const {
      return (long)m_dataStack.depth();
   }

   inline long localVarCount() const {
      return (long)m_dataStack.depth() - currentFrame().m_stackBase;
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
      return *m_dataStack.addSlot();
   }

   inline bool dataEmpty() const { return m_dataStack.m_top < m_dataStack.m_base; }

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
      int64 vals = params()[-1].asInteger();

      // roll back to previous state stack state.
      m_dataStack.m_top = m_dataStack.m_base + callf.m_stackBase - 2;
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
      int64 vals = params()[-1].asInteger();
      while( (vals & 0xFFFFFFFF) != 0xFFFFFF )
      {
         m_dataStack.m_top = m_dataStack.m_base + callf.m_stackBase - 2;

         // roll back to previous state stack state.
         callf.m_stackBase = (int32) (vals >> 32);

         vals = params()[-1].asInteger();
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
      int32 baseRuleTop = params()[-1].content.mth.ruleTop;
      // move forward the stack base.
      cf.m_stackBase = baseRuleTop;
      m_dataStack.m_top = m_dataStack.m_base + baseRuleTop + localCount - 1;

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
         register Item& itm = m_dataStack.m_base[cf.m_stackBase-1];
         itm.setOob(true);
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
         register Item& itm = m_dataStack.m_base[cf.m_stackBase-1];
         register bool mode = itm.isOob();
         itm.resetOob();
         return mode;
      }
      return false;
   }
   
   //=========================================================
   // Code frame management
   //=========================================================

   const CodeFrame& currentCode() const { return *m_codeStack.m_top; }
   CodeFrame& currentCode() { return *m_codeStack.m_top; }

   inline void popCode() {
      m_codeStack.pop();
      PARANOID( "Code stack underflow", m_codeStack.depth() >= 0 );
   }

   inline void popCode( int size ) {
      m_codeStack.pop(size);
      PARANOID( "Code stack underflow", m_codeStack.depth() >= 0 );
   }

   inline void unrollCode( int size ) {
      m_codeStack.unroll( size );
      PARANOID( "Code stack underflow", m_codeStack.depth() >= 0 );
   }

   /** Changes the currently running pstep.
    *
    *  Other than changing the top step, this method resets the sequence ID.
    *  Be careful: this won't work for PCode (they need the seq ID to be set to their size).
    */
   inline void resetCode( const PStep* step ) {
      CodeFrame& frame = currentCode();
      frame.m_step = step;
      frame.m_seqId = 0;
   }
   
   inline void resetAndApply( const PStep* step ) {
      CodeFrame& frame = currentCode();
      frame.m_step = step;
      frame.m_seqId = 0;
      step->apply( step, this );
   }

   /** Returns the current code stack size.
    *
    * During processing of SynTree and CompExpr, the change of the stack size
    * implies the request of a child item for the control to be returned to the VM.
    *
    */
   inline long codeDepth() const { return (long)m_codeStack.depth(); }

   /** Push some code to be run in the execution stack.
    *
    * The step parameter is owned by the caller.
    */
   inline void pushCode( const PStep* step ) {
      register CodeFrame* cf = m_codeStack.addSlot();
      cf->m_step = step;
      cf->m_seqId = 0;
   }

   /** Push some code to be run in the execution stack, but only if not already at top.
    \param step The step to be pushed.
    
    This methods checks if \b step is the PStep at top of the code stack, and
    if not, it will push it.

    \note The method will crash if called when the code stack is empty.

     */
   inline void condPushCode( const PStep* step )
   {
      fassert( m_codeStack.m_top >= m_codeStack.m_base );
      if( m_codeStack.m_top->m_step != step )
      {
         pushCode( step );
      }
   }

   inline bool codeEmpty() const { return m_codeStack.depth() == 0; }
   
   //=========================================================
   // Call frame management.
   //=========================================================

   /** Returns the self item in the local context.
    */
   inline const Item& self() const { return currentFrame().m_self; }
   inline Item& self() { return currentFrame().m_self; }

   const CallFrame& previousFrame( uint32 n ) const { return *(&currentFrame()-n); }
   
   const CallFrame& currentFrame() const { return *m_callStack.m_top; }
   CallFrame& currentFrame() { return *m_callStack.m_top; }

   const Item& regA() const { return m_regA; }
   Item& regA() { return m_regA; }

   inline long callDepth() const { return (long)m_callStack.depth(); }

   inline CallFrame* addCallFrame()  {
      return m_callStack.addSlot();
   }

   /** Prepares a new methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams, const Item& self )
   {
      register CallFrame* topCall = m_callStack.addSlot();
      topCall->m_function = function;
      topCall->m_closedData = 0;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      // TODO: enable rule application with dynsymbols?
      topCall->m_dynsBase = m_dynsStack.depth();
      topCall->m_paramCount = nparams;
      topCall->m_self = self;
      topCall->m_bMethodic = true;
      topCall->m_finallyCount = 0;

      return topCall;
   }

   /** Prepares a new non-methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams )
   {
      register CallFrame* topCall = m_callStack.addSlot();
      topCall->m_function = function;
      topCall->m_closedData = 0;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      // TODO: enable rule application with dynsymbols?
      topCall->m_dynsBase = m_dynsStack.depth();
      
      topCall->m_paramCount = nparams;
      topCall->m_self.setNil();
      topCall->m_bMethodic = false;
      topCall->m_finallyCount = 0;
      
      return topCall;
   }
   
   /** Prepares a new non-methodic closure call frame. */
   inline CallFrame* makeCallFrame( Function* function, ItemArray* cd, int nparams )
   {
      register CallFrame* topCall = m_callStack.addSlot();
      topCall->m_function = function;
      topCall->m_closedData = cd;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      // TODO: enable rule application with dynsymbols?
      topCall->m_dynsBase = m_dynsStack.depth();
      topCall->m_paramCount = nparams;
      topCall->m_self.setNil();
      topCall->m_bMethodic = false;
      topCall->m_finallyCount = 0;

      return topCall;
   }

   bool isMethodic() const { return currentFrame().m_bMethodic; }

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

    The effect of this function is that of popping \b count items from the
    stack and then pushing the \b result. If \b count is 0, \b result is just
    pushed at the end of the stack.
    
    \note this method may be used also by pseudofunctions and generically
    by any PStep in need to access the top of the stack.
    */
   inline void stackResult( int count, const Item& result )
   {
      if( count > 0 )
      {
         popData( count-1 );
         topData() = result;
      }
      else
      {
         pushData( result );
      }
   }

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

   
   /** Called by a calling method to know if the called sub-methdod required a deep operation.
    \return true if the sub-method needed to go deep.

    A function that calls some code which might eventually push some other
    PStep and require virtual machine interaction need to check if this
    actually happened after it gets in control again.

    To cleanly support this operation, a caller should:
    # Push its own callback-pstep
    # Call the code that might push its own psteps
    # Check if its PStep is still at top code frame. If not, return immediately
    # When done, pop its PStep

    For instance:
    \code
    void SomeClass::op_something( VMContex* ctx, ... )
    {
      ...
       ctx->pushCode( &SomeClass::m_myNextStep );
       someItem->op_somethingElse( ctx, .... );
       if ( ctx->wentDeep( &SomeClass::m_myNextStep ) )
       {
         return;
       }
       ...
       ctx->popCode();
    }
    \endcode

    @note This method is subject to false negative if something pushed
    the same pstep that is checked. Use wentDeepSized in case this could
    happen.
    */
   inline bool wentDeep( const PStep* top )
   {
      return m_codeStack.m_top->m_step != top;
   }
   
   inline bool wentDeepSized( long depth )
   {
      return depth != (m_codeStack.m_top - m_codeStack.m_base)+1;
   }
   
   /** Applies a pstep by pushing it and immediately invoking its apply method.
    \param ps The step to be applied.
    
    The applied pstep has now the change to remove itself if its work is done,
    stay pushed to continue its work at a later time or push other psteps that
    need to be executed afterwards (with or without removing itself in the
    process).
    */
   inline void stepIn( const PStep* ps ) {      
      pushCode( ps );
      ps->apply( ps, this );      
   }
   
   /** Step in and check if the caller should yield the control.
    \param ps The step to be performed.
    \return True if the caller should immediately return, false otherwise.
    
    This method invoke the required PStep apply and signals to the caller
    if it should yield the control back upstream (possibly to the Virtual Machine)
    or if it COULD continue performing further operations.

    */
   inline bool stepInYield( const PStep* ps ) {
      const CodeFrame* top = m_codeStack.m_top;
      pushCode( ps );
      ps->apply( ps, this );
      return top != m_codeStack.m_top || m_event != eventNone;
   }
   
   /** Step in and check if the caller should yield the control (optimized).
    \param ps The step to be performed.
    \param top The topmost codeframe when this method is called.
    \return True if the caller should immediately return, false otherwise.
    
    This method invoke the required PStep apply and signals to the caller
    if it should yield the control back upstream (possibly to the Virtual Machine)
    or if it COULD continue performing further operations.

    This versions of stepInYeld uses a previously fetched top code frame
    to perform the check so that it can be used multiple times in tight loops.
    
    */
   inline bool stepInYield( const PStep* ps, const CodeFrame& top ) {
      pushCode( ps );
      ps->apply( ps, this );
      return &top != m_codeStack.m_top || m_event != eventNone;
   }
   
   /** Step in and check if the caller should yield the control (optimized).
    \param ps The step to be performed.
    \param depth Current depth of the code stack.
    \return True if the caller should immediately return, false otherwise.
    
    This method invoke the required PStep apply and signals to the caller
    if it should yield the control back upstream (possibly to the Virtual Machine)
    or if it COULD continue performing further operations.

    This versions of stepInYeld uses a previously fetched stack depth
    to perform the check so that it can be used multiple times in tight loops.
    
    */
   inline bool stepInYield( const PStep* ps, long depth ) {
      pushCode( ps );
      ps->apply( ps, this );
      return (depth != (m_codeStack.m_top-m_codeStack.m_base+1)) || m_event != eventNone;
   }

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

   /** Pushes a "context complete" marker on the stack.
    @see setComplete
    */
   void pushComplete();

  
   /** Prepares the VM to execute a function.

    The VM gets readied to execute the function from the next step,
    which may be invoked via step(), run() or by returning from the caller
    of this function in case the caller has been invoked by the VM itself.
    
    \note At function return, np items from the stack will be popped, and the
    return value of the function (or nil if none) will be placed at the top
    of the stack. This means that the item immediately preceeding the first
    parameter will be overwritten. The call expressions generate the item containing
    the callable entity at this place, but entities invoking functions outside
    call expression evaluation contexts must be sure that they have an extra
    space in the stak where the return value can be placed. In doubt,
    consider using callItem which gives this guarantee.
    
    @param function The function to be invoked.
    @param np Number of parameters that must be already in the stack.
    */ 
   void call( Function* f, int np );

   /** Prepares the VM to execute a function (actually, a method).

    The VM gets readied to execute the function from the next step,
    which may be invoked via step(), run() or by returning from the caller
    of this function in case the caller has been invoked by the VM itself.
    
    This version of call causes the current frame to be considered as
    "methodic", that is, to have a valid "self" item and a method bit marker
    set in the context. This helps to select different behavior in functions
    that may be invoked directly or as a mehtod for various items.
    
    \note At function return, np items from the stack will be popped, and the
    return value of the function (or nil if none) will be placed at the top
    of the stack. This means that the item immediately preceeding the first
    parameter will be overwritten. The call expressions generate the item containing
    the callable entity at this place, but entities invoking functions outside
    call expression evaluation contexts must be sure that they have an extra
    space in the stak where the return value can be placed. In doubt,
    consider using callItem which gives this guarantee.
    
    @param function The function to be invoked.
    @param np Number of parameters that must be already in the stack.
    @param self The item on which this method is invoked. Pure functions are
                considered methods of "nil".
    */
   void call( Function* function, int np, const Item& self );

   /** Invokes a function passing closure data. 
    \see ClassClosure
    */
   void call( Function* function, ItemArray* closedData, int nparams );
   
   /** Calls an item without parameters.
    \see callItem( const Item&, ... )
    */
   void callItem( const Item& item ) { callItem( item, 0, 0 ); }
   
   /** Calls an arbitrary item with arbitrary parameters.
    \param item The item to be called.
    \throw Non-callable Error if this item doesn't provide a valid Class::op_call
    callback.
   
    This method prepares the invocation of an item as if it was called by
    the script. The method can receive variable parameters of type Item&, 
    which must exist \b outside the stack of this context (that might be
    destroyed during the peparation of this call).
    
    This method pushes the called item and the parameters up to when it meets
    a 0 in the parameter list, then invokes the Class::op_call mehtod of the
    class of \b item.
    
    The item may be a method, a function a functor or any other callable entity,
    that is, any item whose Class has op_call reimplemented.
    
    After this method, the caller should immediately retour, as the method might
    immediately invoke the required code if possible, or prepare it for later
    invocation by the VM if not possible. In both cases, the context stacks
    and status must be considered invalidated after this method returns.
    
    \note The variable parameters must be Item instances whose existence must
    be ensured until callItem() method returns. The contents of the items must 
    either be static and exist for the whole duration of the program or be 
    delivered to the Garbage Collector for separate accounting.
    
    Example usage:
    \code
    void somefunc( VMContext* ctx, const Item& callable )
    {
       ....
       Item params[3] = { Item(10), Item( "Hello world" ), Item( 3.5 ) };
       ctx->callItem( item, 3, params );    
    }
    \endcode
    */
   void callItem( const Item& item, int pcount, Item const* params );
   
   /** Returns from the current frame.
    \param value The value considered as "exit value" of this frame.

    This methods pops che call stack of 1 unit and resets the other stacks
    to the values described in the call stack. Also, it sets the top data item
    after stack reset to the "return value" passed as parameter.
    */
   void returnFrame( const Item& value = Item() );
  
      
   /** Unrolls the code and function frames down to the nearest "next" loop base. 
    \throw CodeError if base not found.
    */
   void unrollToNextBase();

   /** Unrolls the code and function frames down to the nearest "break" loop base. 
    \throw CodeError if base not found.
    */
   void unrollToLoopBase();
   
   /** Unrolls the code and function frames down to the nearest "safe" code. 
    \return false if the safe code is not found.
    
    If safe code is not found, the caller should consider discarding the context
    or resetting it.
    
    \see setSafeCode();
    */
   bool unrollToSafeCode();
   
   bool ruleEntryResult() const { return m_ruleEntryResult; }
   void ruleEntryResult( bool v ) { m_ruleEntryResult = v; }
   
   //=============================================================
   // Try/catch
   //   
   
   /** Called back on item raisal.
    \param raised The item that was explicitly raised.
    
    This method searches the try-stack for a possible catcher. If one is found,
    the catcher is activated, otherwise an uncaucght raise error is thrown
    at C++ level.
    
    \note This method will unbox items containing instances of subclasses of 
    ClassError class and pass them to manageError automatically. However, 
    unboxing won't happen for script classes derived from error classes.
    
    \see manageError()
    */
   void raiseItem( const Item& raised );
   
   /** Tries to manage an error through try handlers.
    \param exc The Falcon::Error instance that was thrown.
    
    This method is invoked by the virtual machine when it catches an Error*
    thrown by some part of the code. This exceptins are usually meant to be
    handled by a script, if possible, or forwarded to the system if the
    script can't manage it.
    
    If an handler is found, the stacks are unrolled and the execution of
    the error handler is preapred in the stack. 
    
    If a finally block is found before an error handler can be found, the
    error is stored in that try-frame and the cleanup procedure is invoked
    by unrolling the stacks up to that point; a marker is left in the finalize
    pstep frame so that, after cleanup, the process can proceed.
    
    If a new error is thrown during a finalization, two things may happen: either
    it is handled by an handler inside the finalize process or it is unhandled.
    If it is handled, then the error raisal process continues. If it is unhandled,
    the new error gets propagated as a sub-error of the old one. Example:
    
    @code
    try
      raise Error( 100, "Test Error" )
    finally
      raise Error( 101, "Test From Finally" )
    end
    @endcode
    
    In this case, the error 101 becomes a sub-error of error 100, and the
    error-catching procedure continues for error 100.
    
    If a non-error item is raised from finally, it becomes an sub-error of the
    main one as a "uncaught raised item" exception. If the non-error item was
    raised before an error or an item throw by finally, the non-error item is
    lost and the raisal process continues with the item raised by the finally
    code.
    */
   void raiseError( Error* exc );

   /** Proceeds after a finally frame is complete.
    Invoked by cleanup frames after a finally block has been invoked.
    
    The context leaves a marker in the sequence ID of the cleanup step; if found,
    the cleanup step invoke this method, which pops the current try-frame and
    repeats the try-unroll procedure using the error that was saved in the frame.
    */
   void finallyComplete();
   
   /** Increments the count of traversed finally blocks in this frame.
    
    When a return is issued from inside a try block having a finally clause,
    the return frame unroll procedure must first respect the finally block and
    execute it. As unrolling the stack in search for the finally block is
    more complex than simply executing a flat unroll, having the count of
    active finally blocks helps to preform finally unrolls only when needed.
    
    This method is called when the code generator places a finally block on
    the code stack, and its effects are reversed by enterFinally(), which
    is invoked at the beginning of the finally block.
    
    */
   void traverseFinally() { ++currentFrame().m_finallyCount; }
   
   /** Declares the begin of the execution of a finally block.
    \see traverseFinally
    */
   void enterFinally() { --currentFrame().m_finallyCount; }
   
   /** Finally continuation mode type. */
   typedef enum
   {
      /** Nothing to continue after finally */
      e_fin_none,
      /** Finally called during error raisal. */
      e_fin_raise,
      /** Finally called during a loop break. */
      e_fin_break,
      /** Finally called during a loop continue. */
      e_fin_continue,
      /** Finally called during a return stack unroll. */
      e_fin_return,
      /** Finally called during an explicit and soft termination request. */
      e_fin_terminate
   }
   t_fin_mode;

   void setFinallyContinuation( t_fin_mode fm ) { m_finMode = fm; }
   
   void setCatchBlock( const SynTree* ps ) { m_catchBlock = ps; }

//==========================================================
// Status management
//

   /** Events that may cause VM suspension or interruption. */
   typedef enum {
      /** All green, go on. */
      eventNone,
      /** Debug Breakpoint reached -- return. */
      eventBreak,
      /** Explicit return frame hit -- return. */
      eventReturn,
      /** Quit request -- get out. */
      eventTerminate,
      /** All the code has been processed -- get out. */
      eventComplete,
      /** Soft error raised -- manage it internally if possible. */
      eventRaise
   } t_event;

   /** Returns the last event that was raised in this VM.
      @return The last event generated by the vm.
    */
   inline t_event event() const { return m_event; }   

   /** Asks for a light termination of the VM.
    The VM immediately returns to the caller. The event is sticky; this means
    that intermediate callers will see this event set and propagate it upwards.
    */
   inline void quit() { m_event = eventTerminate; }

   /** Asks the VM to exit from its main loop.

    This is generally handled by a specific opcode that asks for the VM to
    exit its current main loop. The opcode may be inserted by special
    "atomic" calls. When such a call is made, if the called function needs
    to ask the VM to perform some calculation, then it can add this opcode
    to the code stack to be called back when the calculation is done.

    This is meant to be used internally by the engine and shouldn't be
    tackled by modules or external code.

    To cause the code flow to be temporarily suspended at current point
    you may use the pushBreak() or pushReturn() methods, or otherwise push
    your own PStep taking care of breaking the code.
    */
   inline void setReturn() { m_event = eventReturn; }

   /** Resets the event.
   */
   inline void clearEvent() { m_event = eventNone; }

   /** Sets the complete event
   */
   inline void setComplete() { m_event = eventComplete; }

   /** Activates a breakpoint.

      Breaks the current run loop. This is usually done by specific
      breakpoint opcodes that are inserted at certain points in the code
      to cause interruption for debug and inspection.

    To cause the code flow to be temporarily suspended at current point
    you may use the pushBreak() or pushReturn() methods, or otherwise push
    your own PStep taking care of breaking the code.

    The StmtBreakpoint can be inserted in source flows for this purpose.
   */
   inline void breakpoint() { m_event = eventBreak; }
   
   Error* thrownError() const { return m_thrown; }
   Error* detachThrownError() { Error* e = m_thrown; m_thrown =0; return e; }
   
   /** Check the boolean true-ness of the topmost data item, possibly going deep.
    */
   bool boolTopData();
   
   /** Check the boolean true-ness of the topmost data item, removing the top element.
    */
   inline bool boolTopDataAndPop()
   {
      bool btop = boolTopData();
      popData();
      return btop;
   }
   
   //===============================================================
   // Dynamic Symbols
   //
   /** Sets the value associated with a dynamic symbol.
    \param name the name of the dynsymbol to be associated.
    \param value The new value to be associated with the symbol.
    
    If the symbol exists in the local context, it is updated. If it does
    not exist, it is created and the value is stored as the symbol item.
    
    \note The value is set through Item::assign, respecting referencing and
    copy semantics.
    */
   void setDynSymbolValue( DynSymbol* dyns, const Item& value );
   
   /** Gets the the value associated with a dynamic symbol.
    \param name the name of the dynsymbol to be associated.
    \return A pointer to the item associated with the symbol.
    
    If the symbol exists in the local context, its associated value is returned.
    if it doesn't exist, it is searched through the local context from the current
    function to the topmost function. 
    
    If a local symbol corresponding to the given name is found, its value is
    referenced and the reference item is associated with the name; the referenced
    item (already de-referenced) is returned.
    
    If a local symbol is not found, then the global symbol table of the module
    of the topmost function is searched. If the symbol is found the same operation
    as above is performed.
    
    If the symbol is not found, it is searched in the globally exported symbol
    map in the virtual machine. Again, if the search is positive, it is 
    associated as per above.
    
    If the search finally fails, a NIL item is created in the local dynsymbol
    context and associated to the symbol, and that is returned.
    
    \note Symbols marked as constant are returned by value; they aren't referenced.
    */
   Item* getDynSymbolValue( DynSymbol* dyns );
   
protected:
   
   /** Class holding the dynamic symbol information on a stack. */
   class DynsData {
   public:
      DynSymbol* m_sym;
      Item m_item;
      
      DynsData():
         m_sym(0)
      {}
      
      DynsData( DynSymbol* sym ):
         m_sym(sym)
      {}
      
      DynsData( DynSymbol* sym, const Item& data ):
         m_sym(sym),
         m_item(data)
      {}
      
      DynsData( const DynsData& other ):
         m_sym(other.m_sym),
         m_item(other.m_item)
      {}
      
      ~DynsData() {}      
   };

   template<class datatype__>
   class LinearStack
   {
   public:
      static const int INITIAL_STACK_ALLOC = 512;
      static const int INCREMENT_STACK_ALLOC = 256;

      datatype__* m_base;
      datatype__* m_top;
      datatype__* m_max;

      LinearStack(): m_base(0) {}
      ~LinearStack();

      void init( int base = -1 );
      
      inline void reset( int base = -1)
      {
         m_top = m_base-base;
      }

      inline long height() const {
         return m_top - m_base;
      }

      inline long depth() const {
         return (m_top - m_base)+1;
      }
      
      /** Returns a pointer to a position displaced from base.
       This can be used in conjuction with the base value stored in call frames
       to get the base of the current frame in the stack.
       */
      inline datatype__* offset( long dist )
      {
         return m_base + dist;
      }

      void more();
      
      inline void pop() {
         --m_top;
      }
      
      inline void pop( int count ) {
         m_top -= count;
      }
      
      inline void unroll( int oldSize ) {
         m_top = m_base + oldSize - 1;
      }
      
      inline datatype__* addSlot() {
         ++m_top;
         if( m_top >= m_max )
         {
            more();
         }
         return  m_top;
      }
   };

   // Inner constructor to create subclasses
   VMContext( bool );
   
   LinearStack<CodeFrame> m_codeStack;
   LinearStack<CallFrame> m_callStack;
   LinearStack<Item> m_dataStack;
   LinearStack<DynsData> m_dynsStack;
      
   Item m_regA;
   uint64 m_safeCode;

   /** Error whose throwing was suspended by a finally handling. */
   Error* m_thrown;
   
   /** Item whose raisal was suspended by a finally handling. */
   Item m_raised;
   
   // finally continuation mode.
   t_fin_mode m_finMode;
  
   VMachine* m_vm;

   friend class VMachine;
   friend class SynFunc;
  
   bool m_ruleEntryResult;
   
   // temporary variable used during stack unrolls.
   const SynTree* m_catchBlock;
   
   // last raised event.
   t_event m_event;
   
   template <class __checker> bool unrollToNext( const __checker& check );
   
   /** Declares an unhandled error.
    This method is invoked after an error is unhandled. The virtual machine
    sees the error at run() loop and can handle it to the caller or throw it.
    */
   void unhandledError( Error* error );
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
