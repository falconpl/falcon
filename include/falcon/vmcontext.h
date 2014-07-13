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
#include <falcon/trace.h>
#include <falcon/item.h>
#include <falcon/codeframe.h>
#include <falcon/callframe.h>
#include <falcon/variable.h>
#include <falcon/paranoid.h>
#include <falcon/closure.h>
#include <falcon/locationinfo.h>

#include <falcon/atomic.h>
#include <falcon/refcounter.h>
#include <falcon/process.h>

#include <string.h>

namespace Falcon {

class VMachine;
class SynFunc;
class StmtTry;
class SynTree;
class Symbol;
class Storer;
class SymbolTable;
class Shared;
class Scheduler;
class ContextGroup;
class Process;
class SymbolMap;
class TreeStep;
class GCToken;

/**
 * Execution context for Falcon virtual machine.
 *
 * A pure VM context is preferentially used by the engine
 * and extensions that integrate in the engine.
 *
 * Embedders are directed to use WVMContext that has more
 * support and facilities for third party applications.
 *
 */
class FALCON_DYN_CLASS VMContext
{
public:
   const static int32 evtTerminate = 0x1;
   const static int32 evtComplete = 0x2;
   const static int32 evtBreak = 0x4;
   const static int32 evtSwap = 0x8;
   const static int32 evtRaise = 0x10;
   /** Request all the stepInYield to return the control to the processor **/
   const static int32 evtEmerge = 0x20;

   /** Declare the timeslice expired for this context **/
   const static int32 evtTimeslice = 0x40;

   const static int32 statusBorn = 0x00;
   const static int32 statusReady = 0x01;
   const static int32 statusActive = 0x02;
   const static int32 statusDescheduled = 0x03;
   const static int32 statusInspected = 0x04;
   const static int32 statusSleeping = 0x05;
   const static int32 statusWaiting= 0x06;
   const static int32 statusZombie = 0x07;
   const static int32 statusQuiescent = 0x08;
   const static int32 statusTerminated = 0x10;


   /** Class of callbacks invoked at context termination.
    *
    * This class is used to register entities that should receive
    * a notification about when the context is terminated.
    *
    * The concrete instance is held by the subscriber that
    * registered it with registerWeakRef().
    *
    * If the subscriber goes out of scope and is destroyed
    * before the context is terminated, it must invoke
    * unregisterWeakRef() or the callback process will
    * crash.
    *
    * \note The callback is invoked inside the weak ref list
    * lock. If doing anything lengty there, or holding a mutex,
    * be sure KNOW WHAT YOU'RE DOING. Especially, don't invoke
    * registerWeakRef() or unregisterWeakRef() while  holding
    * any mutex you want to hold also during onTerminate()
    * callback.
    *
    */
   class FALCON_DYN_CLASS WeakRef
   {
   public:
      WeakRef():
         m_bTerminated(false)
      {}

      virtual ~WeakRef() {}

      bool hasTerminated() const { return m_bTerminated; }
      void terminated( VMContext* ctx ) { m_bTerminated = true; onTerminate( ctx ); }

      /**
       * Callback to be overridden, called when the context terminates.
       */
      virtual void onTerminate(VMContext* ctx) = 0;

   private:
      bool m_bTerminated;
      WeakRef* m_next;
      WeakRef* m_prev;

      friend class VMContext;
   };

   VMContext( Process* prc, ContextGroup* grp=0 );

   /** Tells if the context is currently in some active state.
    *
    */
   bool isActive() const { return m_status != statusBorn && m_status != statusTerminated; }

   /**
    Returns the unique ID of this context.

    The ID is unique only within a single virtual machine
    (contexts from different VMs will have clashing IDs),
    and is assigned only after that a context is assigned to a machine.
    \return the context ID in the host VM, or 0 if unassigned.

    */
   uint32 id() const { return m_id; }

   /** Gives a description of the location of the next step being executed.
    @return A string with a textual description of the source position of the
            next step being executed.

    This is a debug function used to indicate where is the next step being
    executed in a source file.
   */
   String location() const;

   /** Gives a description of the location of the next step being executed.
    @param infos An instance of LocationInfo receiving the debug information about
           the location in the source files of the next step.

    This information is more detailed and GUI oriented than the information
    returned by location().
    */
   bool location( LocationInfo& infos ) const;

   /** Outlines VM status in a string.
    @return A string with a textual description of the VM status.

    This is a debug function used to indicate what's the current status of the
    virtual machine.
   */
   String report();

   /** Returns the step that is going to be executed next, or null if none.
    \return The next step that will be executed.
    */
   const PStep* nextStep() const;

   /** Returns the step that is going to be executed next in the given frame, or null if none.
    \param frame the nth calling frame, 0 being the current frame.
    \return The next step that will be executed.

    Calling nextStep(1) will return the step that will be executed when the
    current frame returns (unless it returns a dynamic computation for immediate
    execution).
    */
   const PStep* nextStep( int frame ) const;

   /** Resets the context to the initial state.

    This clears the context and sets it as if it was just created.
    */
   virtual void reset();

   /** Register a callback that is invoked at termination.
    * \see VMContextWeakRef
    * \note The registerer must exist at least up to the moment it receives the termination event.
    */
   void registerOnTerminate( WeakRef* subscriber );

   /** Unregister a previously registered callback at termination.
    * \see VMContextWeakRef
    * \note The registerer must exist at least up to the moment it receives the termination event.
    */
   void unregisterOnTerminate( WeakRef* subscriber );

   /** Registers this context in the garbage collector.
    *
    * Used by the process to register the context before starting it.
    *
    * Can be called multiple times; it will be ignored if the context is already
    * registered.
    *
    * This method should be called after creation and before start,
    * if the caller needs to prepare some GC-sensible data right before
    * starting the context.
    *
    * The method will wait for the garbage collector to actually register
    * the context before to proceed.
    *
    * The collector is not able to start a collection loop until the
    * context reaches an executable state; so creating GC sensible data
    * after registration of a new context with the wait option set to true,
    * and before its start, makes that data intrinsically GC-safe.
    *
    * Also, be sure to hold a newly-registered context outside execution
    * for as little as possible, as no GC activity can take place in that
    * section.
    */
   void registerInGC();

   //=========================================================
   // Varaibles - stack management
   //=========================================================


   /** Return the nth parameter in the local context.
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline const Item* param( uint32 n ) const {
      fassert(m_dataStack.m_base+(n + currentFrame().m_dataBase) < m_dataStack.m_max );
      if( currentFrame().m_paramCount <= n ) return 0;
      return &m_dataStack.m_base[ n + currentFrame().m_dataBase ];
   }

   /** Return the nth parameter in the local context (non-const).
   \param n The parameter number, starting from 0.
   \return A pointer to the nth parameter in the stack, or 0 if out of range.
    */
   inline Item* param( uint32 n )  {
      if( currentFrame().m_paramCount <= n ) return 0;
      fassert(m_dataStack.m_base+(n + currentFrame().m_dataBase) < m_dataStack.m_max );
      return &m_dataStack.m_base[ n + currentFrame().m_dataBase ];
   }

   /** Now alias to param.
    */
   inline Item* paramRef( uint32 n )  {
      return param(n);
   }

  /** Returns the parameter array in the current frame.
    \return An array of items pointing to the top of the local frame data stack.

    This method returns the values in the current topmost data frame.
    This usually points to the first parameter of the currently executed function.
    */
   inline Item* params() {
      return &m_dataStack.m_base[ currentFrame().m_dataBase ];
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
      fassert(m_dataStack.m_base+(n + currentFrame().m_dataBase + currentFrame().m_paramCount) < m_dataStack.m_max );
      return &m_dataStack.m_base[ n + currentFrame().m_dataBase + currentFrame().m_paramCount ];
   }

   inline Item* local( int n ) {
	  fassert(m_dataStack.m_base+(n + currentFrame().m_dataBase + currentFrame().m_paramCount) < m_dataStack.m_max );
      return &m_dataStack.m_base[ n + currentFrame().m_dataBase + currentFrame().m_paramCount ];
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
         Item temp;
         temp.copy(data);
         Item* base = m_dataStack.m_base;
         m_dataStack.more();
         *m_dataStack.m_top = temp;
         onStackRebased( base );
      }
      else {
         m_dataStack.m_top->copy(data);
      }
   }
   inline void pushDataLocked( const Item& data ) {
      ++m_dataStack.m_top;
      if( m_dataStack.m_top >= m_dataStack.m_max )
      {
         // = locks
         Item temp = data;
         Item* base = m_dataStack.m_base;
         m_dataStack.more();
         *m_dataStack.m_top = temp;
         onStackRebased( base );
      }
      else {
         // = locks
         *m_dataStack.m_top = data;
      }
   }

   /** Add more variables on top of the stack.
    \param count Number of variables to be added.
    The resulting values will be nilled.
    */
   inline void addLocals( size_t count ) {
      Item* base = m_dataStack.m_top;
      m_dataStack.m_top += count;
      if( m_dataStack.m_top >= m_dataStack.m_max )
      {
         Item* old = m_dataStack.m_base;
         m_dataStack.more();
         onStackRebased( old );

         base = m_dataStack.m_top - count;
      }

      memset( base+1, 0, count * sizeof(Item) );
   }

   /** Add more variables on top of the stack -- without initializing them to nil.
    \param count Number of variables to be added.

    Now an alias to addLocals
    */
   inline void addSpace( size_t count ) {
      addLocals( count );
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

   void removeData( uint32 pos, uint32 removeSize );

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
      return (long)m_dataStack.depth() - currentFrame().m_dataBase;
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
    \note Call this \b after pushing the StmtRule in the code stack.
       UnrollRule and CommitRule will unroll also the PStep that was the current
       one as this method is called.
    */
   void startRuleFrame();

   /** Adds a new local rule frame with a non-deterministic traceback point.
    */
   void startRuleNDFrame( uint32 tbpoint );

   /** Retract the current rule status and resets the rule frame to its previous state.
    */
   uint32 unrollRuleNDFrame();

   void dropRuleNDFrames();

   /** Retract a whole rule, thus closing it.
    */
   void unrollRule();

   /** Commits a whole rule, thus closing it.
    */
   void commitRule();


   const Item& readInit() const
   {
      return m_initRead;
   }

   void writeInit( const Item& init )
   {
      m_initWrite.copyFromRemote( init );
   }

   void commitInit() {
      m_initRead = m_initWrite;
   }

   void clearInit() {
      m_initRead.setNil();
   }

   void saveInit() {
      pushData(m_initWrite);
   }

   void restoreInit() {
      m_initRead = topData();
      popData();
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
   CodeFrame* codeAt( long pos ) const { return m_codeStack.m_top - pos; }

   /** Push some code to be run in the execution stack.
    *
    * The step parameter is owned by the caller.
    */
   inline void pushCode( const PStep* step ) {
      register CodeFrame* cf = m_codeStack.addSlot();
      cf->m_step = step;
      cf->m_seqId = 0;
#ifndef NDEBUG
      cf->m_dataDepth = 0xFFFFFFFF;
      cf->m_dynsDepth = 0xFFFFFFFF;
#endif
   }

   inline void pushCodeWithUnrollPoint( const PStep* step ) {
      register CodeFrame* cf = m_codeStack.addSlot();
      cf->m_step = step;
      cf->m_seqId = 0;
      cf->m_dataDepth = m_dataStack.depth();
      cf->m_dynsDepth = m_dynsStack.depth();
   }

   /** Save the unroll points on this code frame.
    *
    */
   inline void saveUnrollPoint( CodeFrame& cf ) {
      cf.m_dataDepth = m_dataStack.depth();
      cf.m_dynsDepth = m_dynsStack.depth();
   }

   inline void restoreUnrollPoint() {
      m_dataStack.unroll(currentCode().m_dataDepth);
      m_dynsStack.unroll(currentCode().m_dynsDepth);
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
   /**
    * Return a typeized pointer to the self instance.
    */
   template<typename _T>
      _T tself() { return static_cast<_T>(currentFrame().m_self.asInst()); }

   const CallFrame& previousFrame( uint32 n ) const { return *(&currentFrame()-n); }

   const CallFrame& currentFrame() const { return *m_callStack.m_top; }
   CallFrame& currentFrame() { return *m_callStack.m_top; }

   /** Deprecated: Kept for historic reasons */
   const Item& regA() const { return topData(); }
   /** Deprecated: Kept for historic reasons */
   Item& regA() { return topData(); }
   /** Deprecated: Kept for historic reasons */
   void retval( const Item& i ) { regA() = i; }
   /** Deprecated: Kept for historic reasons */
   void retval() {}

   inline long callDepth() const { return (long)m_callStack.depth(); }
   inline CallFrame& callerFrame( uint32 n ) const { return *(m_callStack.m_top - n); }

   inline CallFrame* addCallFrame()  {
      return m_callStack.addSlot();
   }

   /** Prepares a new methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams, const Item& self, bool bMethodic=true )
   {
      register CallFrame* topCall = m_callStack.addSlot();
      topCall->m_function = function;
      topCall->m_closingData = topCall->m_closure = 0;
      topCall->m_codeBase = codeDepth();
      // initialize also initBase, as stackBase may move
      topCall->m_dataBase = dataSize()-nparams;
      // TODO: enable rule application with dynsymbols?
      topCall->m_dynsBase = m_dynsStack.depth();
      topCall->m_dynDataBase = m_dynDataStack.depth();
      topCall->m_paramCount = nparams;
      topCall->m_self = self;
      topCall->m_bMethodic = bMethodic;
      topCall->m_callerLine = m_callerLine;
      m_callerLine = 0;

      return topCall;
   }

   /** Prepares a new non-methodic call frame. */
   inline CallFrame* makeCallFrame( Function* function, int nparams )
   {
      return makeCallFrame( function, nparams, Item(), false );
   }

   /** Prepares a new non-methodic closure call frame. */
   inline CallFrame* makeCallFrame( Closure* cd, int nparams )
   {
      fassert( cd->closed() );
      CallFrame* topCall = makeCallFrame( cd->closed(), nparams, Item(), false );
      topCall->m_closure = cd->data();
      return topCall;
   }

   bool isMethodic() const { return currentFrame().m_bMethodic; }

   ClosedData* getTopClosedData() const;

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

   inline void getMethodicParams( Item*& p0 )
   {
      if ( isMethodic() ) {
         p0 = &self();
      } else {
         p0 = param(0);
      }
   }

   inline void getMethodicParams( Item*& p0, Item*& p1 )
   {
      if ( isMethodic() ) {
         p0 = &self();
         p1 = param(0);
      } else {
         p0 = param(0);
         p1 = param(1);
      }
   }

   inline void getMethodicParams( Item*& p0, Item*& p1, Item*& p2 )
   {
      if ( isMethodic() ) {
         p0 = &self();
         p1 = param(0);
         p2 = param(1);
      } else {
         p0 = param(0);
         p1 = param(1);
         p2 = param(2);
      }
   }

   inline void getMethodicParams( Item*& p0, Item*& p1, Item*& p2, Item*& p3 )
   {
      if ( isMethodic() ) {
         p0 = &self();
         p1 = param(0);
         p2 = param(1);
         p3 = param(2);
      } else {
         p0 = param(0);
         p1 = param(1);
         p2 = param(2);
         p3 = param(3);
      }
   }

   inline void getMethodicParams( Item*& p0, Item*& p1, Item*& p2, Item*& p3, Item*& p4 )
   {
      if ( isMethodic() ) {
         p0 = &self();
         p1 = param(0);
         p2 = param(1);
         p3 = param(2);
         p4 = param(3);
      } else {
         p0 = param(0);
         p1 = param(1);
         p2 = param(2);
         p3 = param(3);
         p4 = param(4);
      }
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
      return top != m_codeStack.m_top || atomicFetch(m_events);
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
      return &top != m_codeStack.m_top || atomicFetch(m_events);
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
      return (depth != (m_codeStack.m_top-m_codeStack.m_base+1)) || atomicFetch(m_events);
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

    \note This method is meant to be called internally by the VM;
    you can use it if you know-what-you're-doing(TM); otherwise, use
    callItem.

    */
   void callInternal( Function* f, int32 np );

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
    \note This method is meant to be called internally by the VM;
    you can use it if you know-what-you're-doing(TM); otherwise, use
    callItem.
    */
   void callInternal( Function* function, int np, const Item& self );

   /** Invokes a function passing closure data.
    \see ClassClosure

    \note This method is meant to be called internally by the VM;
    you can use it if you know-what-you're-doing(TM); otherwise, use
    callItem.
    */
   void callInternal( Closure* closure, int nparams );

   void callInternal( const Item& item, int np );

   /** Calls an item without parameters.
    \see callItem( const Item&, ... )
    */
   void callItem( const Item& item ) { callItem( item, 0, 0 ); }

   /** Calls an arbitrary item with arbitrary parameters.
    \param item The item to be called.
    \param count Number of parameters passed
    \param params Parameter passed.


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

    \note The item array in the \b callable parameter must be stored
    otuside the scope of the Falcon VM, as the items are copied without
    locking nor marking as-is into the context stack.


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


   /** Prepares the VM to execute a function.
    @param function The function to be invoked.
    @param np Number of parameters in the params array.
    @param params Parameters to be pushed in the vm.

    This method is meant to invoke a function from outside the normal
    VM operation. The context prepares an execution context, properly
    pushing the required elements in the data and call stack, and invokes
    the given function immediately.

    After this method returns, the given function might have exited invoking a
    returnFrame() or still be engaged for further computation at
    later steps.
    */
   void call( Function* f, int32 np = 0, Item const* params=0 );

   /** Invokes a method of a given item.
    @param function The function to be invoked.
    @param self The item on which the method is invoked.
    @param np Number of parameters in the params array.
    @param params Parameters to be pushed in the vm.

    This method is meant to invoke a method from outside the normal
    VM operation. The context prepares an execution context, properly
    pushing the required elements in the data and call stack, and invokes
    the given function immediately.

    After this method returns, the given function might have exited invoking a
    returnFrame() or still be engaged for further computation at
    later steps.
    */
   void call( Function* function, const Item& self, int32 np=0, Item const* params=0 );

   /** Invokes a closure.
    @param closure The closure to be invoked.
    @param np Number of parameters in the params array.
    @param params Parameters to be pushed in the vm.

    This method is meant to invoke a closure from outside the normal
    VM operation. The context prepares an execution context, properly
    pushing the required elements in the data and call stack, and invokes
    the given function immediately.

    After this method returns, the given function might have exited invoking a
    returnFrame() or still be engaged for further computation at
    later steps.
    */
   void call( Closure* closure, int32 np=0, Item const* params=0 );

   /** Adds a local symbol table.
    This creates an empty symbol entry in the symbol stack to store the
    current data depth.
    \param st The symbol table containing the Local Symbols to be added.
    \param pcount The count of the parameters that have been pushed for this frame.
    */
   void addLocalFrame( SymbolMap* st, int pcount );

   /** Returns from the current frame.
    \param value The value considered as "exit value" of this frame.

    This methods pops che call stack of 1 unit and resets the other stacks
    to the values described in the call stack. Also, it sets the top data item
    after stack reset to the "return value" passed as parameter.
    */
   void returnFrame( const Item& value = Item() );

   void returnFrameDoubt( const Item& value = Item() );
   void returnFrameEval( const Item& value = Item() );
   void returnFrameDoubtEval( const Item& value = Item() );

   /** Unrolls the code and function frames down to the nearest "next" loop base.
    \throw CodeError if base not found.
    */
   void unrollToNextBase();

   /** Unrolls the code and function frames down to the nearest "break" loop base.
    \throw CodeError if base not found.
    */
   void unrollToLoopBase();

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
   const Item& raised() const { return m_raised; }

   /** Tries to manage an error through try handlers.
    \param exc The Falcon::Error instance that was thrown.
    \return The same entity that is passed as parameter, so that
       it's possible to use it on the fly.

    This method is invoked by the virtual machine when it catches an Error*
    thrown by some part of the code. This exceptions are usually meant to be
    handled by a script, if possible, or forwarded to the system if the
    script can't manage it.

    If an handler is found, the stacks are unrolled and the execution of
    the error handler is prepared in the stack.

    If a finally block is found before an error handler can be found, the
    error is stored in that try-frame and the cleanup procedure is invoked
    by unrolling the stacks up to that point; a marker is left in the finalize
    pstep frame so that, after cleanup, the process can proceed.

    If a new error is thrown during a finalization, two things may happen: either
    it is handled by an handler inside the finalize process or it is not handled.
    If it is handled, then the error raising process continues. If it is not handled,
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
    lost and the raising process continues with the item raised by the finally
    code.

    @note The method returns the same error it receives; in this way, it is
    possible to create the error directly via new statement as the
    parameter and have it back. Notice that the error gets a reference increment
    when it's sent to raiseError(), so it is necessary to dereference it
    when the caller doesn't need it anymore.

    The extension code can call directly raiseError() whenever it's going
    to return the control to the calling Falcon::Processor as soon as possible.
    When raiseError() returns, this context may have significantly changed,
    but if calling code is able to yield the control back to the Falcon system,
    this might save a useless C++ throw. However, when the code detects an error
    deep down in C++, it should simply throw it and let the Falcon::Processor
    main loop to catch it.

    \note Invoking this method causes exiting from any critical section, that is
    releasing (and signaling) any acquired resource.
    */
   Error* raiseError( Error* exc );


   /** Sets the catch block for the current finally unroll. */
   void setCatchBlock( const SynTree* ps ) { m_catchBlock = ps; }

   /** Unroll dynsymbols pushed for local evaluations.
    \param symBase Number of symbols LEFT in the stack after unroll.
    */
   void unrollLocalFrame( int symBase );

   /** Finds the upstream local evaluation frame and pops it.

    If a local frame is not found up to the next code function frame,
    the operation is silently aborted.

   */
   void exitLocalFrame( bool exec = false );

   //==========================================================
   // Event management
   //


   /** Returns the last event that was raised in this VM.
      @return True if the
    */
   inline bool hadEvent() const { return atomicFetch(m_events) != 0; }

   inline int32 events() const { return atomicFetch(m_events); }

   /** Clear all thread-specific and non-fatal events. */
   void clearEvents() { atomicAnd( m_events, evtBreak ); }

   /** Asks for this context to be terminated asap.
       This event is never reset.
    */
   inline void setTerminateEvent() { atomicOr(m_events, evtTerminate); }

   inline bool isTerminated() const { return (atomicFetch( m_events ) & evtTerminate) != 0; }


   /** Sets the complete event
   */
   inline void setCompleteEvent() { atomicOr(m_events, evtComplete);  }

   /** Sets the emerge event
   */
   inline void setEmergeEvent() { atomicOr(m_events, evtEmerge);  }

   /** Activates a breakpoint.

        Breaks the current run loop. This is usually done by specific
        breakpoint opcodes that are inserted at certain points in the code
        to cause interruption for debug and inspection.

      To cause the code flow to be temporarily suspended at current point
      you may use the pushBreak() or pushReturn() methods, or otherwise push
      your own PStep taking care of breaking the code.

      The StmtBreakpoint can be inserted in source flows for this purpose.
     */
   inline void setBreakpointEvent() { atomicOr(m_events, evtBreak);  }
   inline void clearBreakpointEvent() { atomicAnd(m_events, ~evtBreak);  }

   /** Sets the swap event.
    This event ask the processor to swap the context out as soon as possible.
   */
   inline void setSwapEvent() { atomicOr(m_events, evtSwap);  }

   /** Swaps a context out of execution.
    *
    * This method instructs the processor where the context is running to
    * remove the context and NOT send it to the context manager for later
    * rescheduling.
    *
    * Also, it instructs the process that this context is not "live" anymore.
    *
    * However, the context still exists and is accounted for in the garbage
    * collector; it can be re-started by invoking process::start.
    */
   void swapOut();

   /** Sets the inspect event.
    Indicates that this contexts should be inspected by the garbage collector ASAP.
    The swap event is also set.
   */
   void setInspectEvent();

   /** Declare the timeslice for this context expired.
    *
    * This won't lead to an automatic removal of this context from the processor;
    * the context will be changed only if there is another context immediately ready to run.
    */
   void setTimesliceEvent() { atomicOr(m_events, evtTimeslice);  }

   /**
    * Used by the context manager to communicate that the context is quiescent.
    */
   bool goToSleep();

   /**
    * Used by the context manager to communicate that the context is not quiescent anymore.
    */
   void awake();

   Error* thrownError() const { return m_lastRaised; }
   Error* detachThrownError() { Error* e = m_lastRaised; m_lastRaised =0; return e; }


   /** Check the boolean true-ness of the topmost data item, removing the top element.
    */
   inline bool boolTopDataAndPop()
   {
      bool btop = boolTopData();
      popData();
      return btop;
   }

   /** Check the boolean true-ness of the topmost data item, possibly going deep.
    */
   inline bool boolTopData()
   {

      switch( topData().type() )
      {
      case FLC_ITEM_NIL:
         return false;

      case FLC_ITEM_BOOL:
         return topData().asBoolean();

      case FLC_ITEM_INT:
         return topData().asInteger() != 0;

      case FLC_ITEM_NUM:
         return topData().asNumeric() != 0.0;

      case FLC_ITEM_METHOD:
         return true;

      default:
         topData().asClass()->op_isTrue( this, topData().asInst() );
         if(topData().isBoolean() )
         {
            return topData().asBoolean();
         }
         break;
      }

      return false;
   }

   //===============================================================
   // Dynamic Symbols
   //

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
   Item* resolveSymbol( const Symbol* dyns, bool forAssign );
   Item* resolveSymbol( const String& symname, bool forAssign );
   Item* resolveGlobal( const Symbol* name, bool forAssign );
   Item* resolveGlobal( const String& symname, bool forAssign );

   /** Force the symbol to be defined as required.
    * \param sym The symbol to be defined.
    * \param data The data associated with this symbol.
    *
    * This symbol gets defined in the current context/local frame as
    * the given data pointer. The data must stay valid as long as the
    * current frame is alive (so it should be in the local stack or in
    * the global variable space of the module of the current frame function).
    *
    */
   void defineSymbol( const Symbol* sym, Item* data );

   /** Force the symbol to be defined as nil.
    * \param sym The symbol to be defined.
    */
   void defineSymbol( const Symbol* sym );

   /** Copies pcount parameters from the frame parameters area to the top of the stack. */
   void forwardParams( int pcount );

   /** Gets the storer that is currently performing a serialization. */
   Storer* getTopStorer() const;

   /** Aborts waits and acquisitions being performed by this contexts.
    This method notifies all the shared resources that this context was
    waiting for, or trying to acquire, that we're not waiting anymore for them.

    This might happen because a wait timeout has expired, or because this context
    was actually able to acquire one of the resources.

   \note Invoking this method while the context is ready or running will probably
   result in a disaster. The scheduler only calls this mehtod when the context
   is sleeping, and this might also get called also when the context is de-sceduled
   and approaching destruction; but never while running.
    */
   void abortWaits();

   /** Sets the next schedule time.
    When the context is put at sleep because of explicit wait on resources or
    because of a sleep request, this value is set to the absolute time (in
    milliseconds since epoch) when the context should be runnable again.

    This information is used by the scheduler to find the context in the sleeping
    contexts map.

    \note this value is -1 for the runnable or running contexts, it's 0 for the
    contexts that yield their time but are immediately able to run, and it's
    > now for the contexts that really want to wait before being runnable again.

    \param to Next absolute time when the context wants to run.
    */
   void nextSchedule( int64 to ) {
      m_mtx_sleep.lock();
      m_next_schedule = to;
      m_mtx_sleep.unlock();
   }

   /**
    Returns the next time when a context will be set to runnable.

    \return Next schedule time
    \see nextSchedule(int64)
    */
   int64 nextSchedule() const {
      m_mtx_sleep.lock();
      int64 sched = m_next_schedule;
      m_mtx_sleep.unlock();

      return sched;
   }

   void initWait();
   void addWait( Shared* resource );
   Shared* engageWait( int64 timeout );
   Shared* declareWaits();

   int32 waitingSharedCount() const;

   /** Releases the acquired resource.
    * \return true if the release causes event preemption.
    *
    */
   bool releaseAcquired();

   /** Set by the context manager when a resource is signaled during idle time.
    *
    * The step invoked after a shared resource wait can check if the resource
    * has been acquired via getSignaledResource().
    *
    * \note, this invokes shared->incref(). Do not use 0 as parameter.
    */
   void signaledResource( Shared* shared );

   /** Gets and clear the resource signaled during idle time.
    *
    * After having temporarily abandoned the processor due to a failed
    * engageWait(), the pstep that receives the control after wake up
    * can invoke this method to get the signaled resource.
    *
    * If this method returns 0, then the wait timeout expired without
    * the context being signaled.
    *
    * Notice that calling this method clears the signaled resource, so
    * successive calls will return 0.
    *
    * \note The resource returned by this method might also be set by
    * the manager as the acquired resource, if the resource has acquire
    * semantic.
    *
    * \note the shared resource, if returned, has an extra incref that
    * must be disposed.
    */
   Shared* getSignaledResouce();

   /** Ensures the signaled resource is zero, eventually disposing of it.
    *
    * \see signaledResource.
    */
   void clearSignaledResource();

   /** Ask explicit descheduling for the given time lapse.
    *
    * This automatically frees acquired resources and resets
    * the delayed events.
    */
   void sleep( int64 timeout );

   /** Called by the ContextManager to confirm the acquisition of a shared resource.
    * \param shared the resource to acquire or 0
    * This method is automatically called when the context is descheduled or terminated,
    * or when a resource is acquired after a successful wait.
    *
    * The acquired resource is incref'd and any eventually previously acquired resource
    * is signaled and decref'd
    *
    * Can be called with 0 to perform automatic signal of a previously acquired resource.
    *
    * \note this should usually be called just by the context manager.
    */
   void acquire(Shared* shared);

   /** Returns the acquired resource.
    *
    * While there is a non-zero acquired resource, the context
    * is in a critical section and cannot be implicitly swapped.
    *
    * Any explicit swap operation, as well as explicit signaling of
    * the acquired resource, causes the acquired resource to be released,
    * and the context to exit the critical section.
    *
    * Notice that a direct Shared::signal operation doesn't automatically
    * removes it from the contexts that have acquired it. It's necessary
    * that the script-level operations signaling the resource check for
    * it to be the context acquired resource, and eventually release it
    * via acquire(0).
    *
    * \note VMContext::acquire(0) will automatically signal the acquired
    * resource, so if using this to free the acquired resource from a script-level
    * signal operation, the caller should not re-signal the shared resource.
    *
    */
   Shared* acquired() const { return m_acquired; }

   /** Ask the context to terminate as soon as possible.
    *
    * If the context is currently waiting or sleeping, it is waken up immediately.
    *
    * It is possible that a context being terminated received a wakeup signal,
    * or was managing a critical resource. While the resource is feed, the status
    * of the system could be instable; a terminate request like this should be issued
    * only when the host process is going to be discarded.
    */
   virtual void terminate();

   /** Called back when this context is declared dead.
    */
   virtual void onTerminated();

   /** Called back when this context is declared temporarily done.
    *
    * The base class one just invokes onTerminated.
   */
   virtual void onComplete();

   void inGroup( ContextGroup* grp ) { m_inGroup = grp; }

   ContextGroup* inGroup() const {return m_inGroup;}

   /** The process in which this context moves.
    */
   Process* process() const { return m_process; }

   /** The virtual machine on which the process running this context is on. */
   VMachine* vm() const { return m_process->vm(); }

   /** Adds a finally handler at current code position */
   void registerFinally( TreeStep* finHandler ) {
      FinallyData& dt = *m_finallyStack.addSlot();
      dt.m_depth = codeDepth();
      dt.m_finstep = finHandler;
   }

   void unregisterFinally() {
      m_finallyStack.pop();
   }

   void gcStartMark( uint32 mark );
   virtual void gcPerformMark();
   uint32 currentMark() const { return m_currentMark; }

   bool markedForInspection() const { return m_bInspectMark; }
   void resetInspectEvent() { m_bInspectMark = false; }

   /** Generates a runtime (code) error at current location in the code.
    * \param id Error id that must be generated.
    * \param extra An extra description for the error.
    * \param line The error line, if different from the current code frame PStep line.
    * \return A configured instance of CodeError.
    *
    * Uses the current call and code frame to pinpoint the location of the
    * error; the line can be overridden.
    */
   Error* runtimeError( int id, const String& extra = "", int line = 0 );

   /**
    * Adds information about the currently executed context.
    * \param error The error to be contextualized
    * \param force if true, will cause the context fields in the error to be overwritten
    *    even if they have already a value.
    */
   void contextualize(Error* error, bool force = false);

   /**
    * Stores the trace-back information for the current context.
    */
   void fillTraceBack(TraceBack* tb, bool bRenderParams=true, long maxDepth=-1);

   /** Stack events during critical sections for later honoring.
    *
    * When a processor is asked to swap or timeslice out a context,
    * this request cannot be fulfilled if the context has an active
    * acquired resource (it is said to be in a critical section).
    *
    * As soon as the acquired resource is released, the delayed event must be
    * set and honored.
    */
   void delayEvents( uint32 evt )
   {
      m_suspendedEvents |= evt;
   }

   /**
    * Context status, for debugging purposes.
    *
    * What the context is currently doing in the system.
    */
   int getStatus();

   /**
    * Context status, for debugging purposes.
    *
    * What the context is currently doing in the system.
    */
   void setStatus( int status );

   /**
    * Depth of the dynamic symbol stack
    */
   inline long dynsDepth() const { return m_dynsStack.depth(); }

   /** Class holding the dynamic symbol information on a stack. */
   class DynsData {
   public:
      const Symbol* m_sym;
      Item* m_value;

      DynsData():
         m_sym(0),
         m_value(0)
      {}


      DynsData( const Symbol* sym, Item* value ):
         m_sym(sym),
         m_value(value)
      {}

      DynsData( const DynsData& other ):
         m_sym(other.m_sym),
         m_value(other.m_value)
      {
      }

      ~DynsData() {}
   };

   inline DynsData* dynsAt(long pos) const { return m_dynsStack.m_top - pos; }

   void callerLine( int32 l ) { m_callerLine = l; }

protected:

   inline Item* addDynData( const Item& data = Item() ) {
      ++m_dynDataStack.m_top;
      if( m_dynDataStack.m_top >= m_dynDataStack.m_max )
      {
         Item temp;
         temp.copy(data);
         Item* base = m_dynDataStack.m_base;
         m_dynDataStack.more();
         *m_dynDataStack.m_top = temp;
         onDynStackRebased( base );
      }
      else {
         m_dynDataStack.m_top->copy(data);
      }

      return m_dynDataStack.m_top;
   }

   /** Class holding registered finally points */
   class FinallyData {
   public:
      const TreeStep* m_finstep;
      uint32 m_depth;

      FinallyData()
      {}

      FinallyData( TreeStep* fs, uint32 d ):
         m_finstep(fs),
         m_depth(d)
      {}
   };

   template<class datatype__>
   class FALCON_DYN_CLASS LinearStack
   {
   public:
      static const int INITIAL_STACK_ALLOC = 256;
      static const int INCREMENT_STACK_ALLOC = 256;
      uint32 m_allocSize;

      datatype__* m_base;
      datatype__* m_top;
      datatype__* m_max;

      LinearStack(): m_allocSize(0), m_base(0), m_top(0), m_max(0) {}
      ~LinearStack();

      void init( int base = -1 );
      void init( int base, uint32 allocSize );

      inline void reset( int base = -1)
      {
         m_top = m_base+base;
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

      void more()
      {
         long distance = (long)(m_top - m_base);
         long newSize = (long)(m_max - m_base + INCREMENT_STACK_ALLOC);
         TRACE("Reallocating %p: %d -> %ld", m_base, (int)(m_max - m_base), newSize );

         datatype__* mem = (datatype__*) realloc( m_base, newSize * sizeof(datatype__) );
         if( mem == 0 )
         {
            throw "No more mem";
         }
         m_base = mem;
         m_top = m_base + distance;
         m_max = m_base + newSize;
         // More is done after moving top, but before using it.
         memset( m_top, 0, (m_max - m_top) * sizeof(datatype__) );
      }


      inline void pop() {
         --m_top;
      }

      inline void pop( int count ) {
         m_top -= count;
      }

      inline void unroll( int oldSize ) {
         m_top = m_base + (oldSize - 1);
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

   atomic_int m_status;
   int32 m_callerLine;

   LinearStack<CodeFrame> m_codeStack;
   LinearStack<CallFrame> m_callStack;
   LinearStack<Item> m_dataStack;
   LinearStack<Item> m_dynDataStack;
   LinearStack<DynsData> m_dynsStack;
   LinearStack<FinallyData> m_finallyStack;

   // list of variables we're waiting on
   LinearStack<Shared*> m_waiting;
   // single variable acquired.
   Shared* m_acquired;
   Shared* m_signaledResource;

   /** Error that was last raised in the context. */
   Error* m_lastRaised;

   /** Item whose raisal was suspended by a finally handling. */
   Item m_raised;

   Item m_initRead;
   Item m_initWrite;

   friend class VMachine;
   friend class SynFunc;
   friend class Scheduler;

   // temporary variable used during stack unrolls.
   const SynTree* m_catchBlock;

   typedef enum {
      e_unroll_not_found,
      e_unroll_found,
      e_unroll_suspended
   } t_unrollResult;

   template <class __checker> t_unrollResult unrollToNext( const __checker& check );

   uint32 m_id;
   uint32 m_currentMark;
   int64 m_next_schedule;
   bool m_bInspectMark;
   bool m_bSleeping;
   mutable Mutex m_mtx_sleep;

   /** Set whenever an event was activated. */
   atomic_int m_events;
   int32 m_suspendedEvents;

   ContextGroup* m_inGroup;
   Process* m_process;

   Mutex m_mtxWeakRef;
   WeakRef* m_firstWeakRef;

   /** Clear the waits, but doesn't send the shared resources a request to remove this context.
    *
    * This is used by engageWait on early success, prior telling the resources they're being
    * watched.
    */
   void clearWaits();

   virtual ~VMContext();
private:

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(VMContext)

   void onStackRebased( Item* oldBase );
   void onDynStackRebased( Item* oldBase );
   void pushBaseElements();

   template<class _returner>
   void returnFrame_base( const Item& value );

   atomic_int m_registeredInGC;
};

}

#define FALCON_POPCODE_CONDITIONAL_WITH_SEQID( __ctx__, __pstep__, __testcode__ ) \
   {int __seqID__= __ctx__->currentCode().m_seqId;\
   try { __ctx__->popCode(); __testcode__; }\
   catch(...) { ctx->pushCode(__pstep__); __ctx__->currentCode().m_seqId = __seqID__; throw; }}

#define FALCON_POPCODE_CONDITIONAL( __ctx__, __pstep__, __testcode__ )\
   try {__ctx__->popCode(); __testcode__; }\
   catch(...) { ctx->pushCode(__pstep__); throw; }


#endif /* FALCON_VMCONTEXT_H_ */

/* end of vmcontext.h */
