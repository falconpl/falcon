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
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline const Item* param( uint32 n ) const {
      fassert(m_dataStack+(n + m_topCall->m_stackBase) < m_maxData );
      if( m_topCall->m_paramCount <= n ) return 0;
      return &m_dataStack[ n + m_topCall->m_stackBase ];
   }

   /** Return the nth parameter in the local context (non-const).
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline Item* param( uint32 n )  {
      fassert(m_dataStack+(n + m_topCall->m_stackBase) < m_maxData );
      if( m_topCall->m_paramCount <= n ) return 0;
      return &m_dataStack[ n + m_topCall->m_stackBase ];
   }

  /** Returns the parameter array in the current frame.
    \return An array of items pointing to the top of the local frame data stack.

    This method returns the values in the current topmost data frame.
    This usually points to the first parameter of the currently executed function.
    */
   inline Item* params() {
      return &m_dataStack[ m_topCall->m_stackBase ];
   }

   inline int paramCount() {
      fassert( m_topCall >= m_callStack );
      return m_topCall->m_paramCount;
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

   /** Push data on top of the stack.
    \item data The data that must be pushed in the stack.
    \note The data pushed is copied by-value in a new stack element, but it is
          not colored.
    */
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

   /** Add more variables on top of the stack -- without initializing them to nil.
    \param count Number of variables to be added.

    This is like addLocals, but doesn't nil the newly created variables.
    */
   inline void addSpace( size_t count ) {
      m_topData += count;
      if( m_topData >= m_maxData )
      {
         moreData();
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
      // Notice: an empty data stack is an error.
      PARANOID( "Data stack underflow", (m_topData >= m_dataStack) );
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
         register Item& itm = m_dataStack[cf.m_stackBase-1];
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

   /** Push some code to be run in the execution stack, but only if not already at top.
    \param step The step to be pushed.
    
    This methods checks if \b step is the PStep at top of the code stack, and
    if not, it will push it.

    \note The method will crash if called when the code stack is empty.

     */
   inline void condPushCode( const PStep* step )
   {
      fassert( m_topCode >= m_codeStack );
      if( m_topCode->m_step != step )
      {
         pushCode( step );
      }
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

   /** Prepares a new methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams, const Item& self )
   {
      register CallFrame* topCall = addCallFrame();
      topCall->m_function = function;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      topCall->m_paramCount = nparams;
      topCall->m_self = self;
      topCall->m_bMethodic = true;

      return topCall;
   }

   /** Prepares a new non-methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams )
   {
      register CallFrame* topCall = addCallFrame();
      topCall->m_function = function;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_initBase = topCall->m_stackBase = dataSize()-nparams;
      topCall->m_paramCount = nparams;
      topCall->m_self.setNil();
      topCall->m_bMethodic = false;

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

    */
   inline bool wentDeep( const PStep* top )
   {
      return m_topCode->m_step != top;
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
     
   template <class __checker> void unrollToNext( const __checker& check );
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
