/*
   FALCON - The Falcon Programming Language.
   FILE: flc_vmcontext.h

   Virtual Machine coroutine execution context.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Virtual Machine coroutine execution context.
*/

#ifndef flc_vmcontext_H
#define flc_vmcontext_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/genericvector.h>
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>
#include <falcon/livemodule.h>
#include <falcon/stackframe.h>

namespace Falcon {

class Symbol;
class Item;
class VMSemaphore;

/** Class representing a coroutine execution context. */
class FALCON_DYN_CLASS VMContext: public BaseAlloc
{
   Item m_regA;
   Item m_regB;

   //Item m_regS1;
   Item m_regL1;
   Item m_regL2;
   Item m_regBind;
   Item m_regBindP;

   VMSemaphore *m_sleepingOn;

   /** Currently executed symbol.
      May be 0 if the startmodule has not a "__main__" symbol;
      this should be impossible when things are set up properly.
   */
   const Symbol* m_symbol;

   /** Module that contains the currently being executed symbol. */
   LiveModule *m_lmodule;

   /** Point in time when this context is due to run again.
    (this is an absolute measure).
   */
   numeric m_schedule;

   int32 m_priority;

   /** Program counter register.
      Current execution point in current code.
   */
   uint32 m_pc;

   /** Next program counter register.
      This is the next instruction the VM has to execute.
      Call returns and jumps can easily modify the VM execution flow by
      changing this value, that is normally set just to m_pc + the length
      of the current instruction.
   */
   uint32 m_pc_next;

   /** Topmost try frame handler.
    Inside a frame, the m_prevTryFrame points to the previous try frame, and
    m_tryPos points to the position in the stack of the current frame where
    the try try is to be restored.
   */
   StackFrame* m_tryFrame;

   /** In atomic mode, the VM refuses to be kindly interrupted or to rotate contexts. */
   bool m_atomicMode;

   friend class VMSemaphore;

   /** Stack of stack frames.
    * The topmost stack frame is that indicated here.
    */
   StackFrame *m_frames;
   StackFrame *m_spareFrames;

public:
   VMContext();
   VMContext( const VMContext& other );
   ~VMContext();

   /** Wakes up the context after a wait. */
   void wakeup( bool signaled = false );

   void priority( int32 value ) { m_priority = value; }
   int32 priority() const { return m_priority; }

   void schedule( numeric value ) { m_schedule = value; }
   numeric schedule() const { return m_schedule; }

   /** Schedule this context after some time from now.
      This function add the current system time to secs and
      prepares this context to be scheduled after that absolute
      time.

      @param secs Number of seconds and fraction after current time.
   */
   void scheduleAfter( numeric secs );

   /** Return true if this is waiting forever on a semaphore signal */
   bool isWaitingForever() const { return m_sleepingOn != 0 && m_schedule < 0; }

   VMSemaphore* waitingOn() const { return m_sleepingOn; }

   void waitOn( VMSemaphore* sem, numeric value=-1 );
   void signaled();

   //===========================
   uint32& pc() { return m_pc; }
   const uint32& pc() const { return m_pc; }

   uint32& pc_next() { return m_pc_next; }
   const uint32& pc_next() const { return m_pc_next; }

   StackFrame* tryFrame() const { return m_tryFrame; }

   Item &regA() { return m_regA; }
   const Item &regA() const { return m_regA; }
   Item &regB() { return m_regB; }
   const Item &regB() const { return m_regB; }

   Item &regBind() { return m_regBind; }
   const Item &regBind() const { return m_regBind; }
   /*
   Item &regBind() { return currentFrame()->m_binding; }
   const Item &regBind() const { return currentFrame()->m_binding; }
   */
   Item &regBindP() { return m_regBindP; }
   const Item &regBindP() const { return m_regBindP; }

   Item &self() { return currentFrame()->m_self; }
   const Item &self() const { return currentFrame()->m_self; }


   /** Latch item.
      Generated on load property/vector instructions, it stores the accessed object.
   */
   const Item &latch() const { return m_regL1; }
   /** Latch item.
      Generated on load property/vector instructions, it stores the accessed object.
   */
   Item &latch() { return m_regL1; }

   /** Latcher item.
      Generated on load property/vector instructions, it stores the accessor item.
   */
   const Item &latcher() const { return m_regL2; }
   /** Latcher item.
      Generated on load property/vector instructions, it stores the accessor item.
   */
   Item &latcher() { return m_regL2; }


   //===========================================

   const ItemArray& stack() const { return m_frames->stack(); }
   ItemArray& stack() { return m_frames->stack(); }

   VMSemaphore *sleepingOn() const { return m_sleepingOn; }
   void sleepOn( VMSemaphore *sl ) { m_sleepingOn = sl; }

   /** Returns the current module global variables vector. */
   ItemArray &globals() { return m_lmodule->globals(); }

   /** Returns the current module global variables vector (const version). */
   const ItemArray &globals() const { return m_lmodule->globals(); }

   /** Returns the currently active live module. */
   LiveModule *lmodule() const { return m_lmodule; }

   /** Changes the currently active live module. */
   void lmodule(LiveModule *lm) { m_lmodule = lm; }

   /** Returns the currently active symbol. */
   const Symbol *symbol() const { return m_symbol; }

   /** Changes the currently active symbol. */
   void symbol( const Symbol* s ) { m_symbol = s; }

   /** Returns the current code. */
   byte* code() const {
      fassert( symbol()->isFunction() );
      return symbol()->getFuncDef()->code();
   }

   /** The currently active frame in this context */
   StackFrame* currentFrame() const
   {
      return m_frames;
   }

   void setFrames( StackFrame* newTop )
   {
      m_frames = newTop;
   }

   /** Creates a stack frame taking a certain number of parameters.
      The stack is created so that it is ready to run the new context;
      use addFrame to add it at a later moment, or prepareFrame to create
      it and add it immediately.

      \param paramCount number of parameters in the stack
      \param frameEndFunc Callback function to be executed at frame end
      \return The newly created or recycled stack frame.
   */
   StackFrame* createFrame( uint32 pcount, ext_func_frame_t frameEndFunc = 0 );

   /** Creates a stack frame and adds it immediately to the stack list.

      This call is equivalent to addFrame( callFrame( pcount, frameEndFunc) ).

      \param paramCount number of parameters in the stack
      \param frameEndFunc Callback function to be executed at frame end
      \return The newly created or recycled stack frame.
   */
   inline StackFrame* prepareFrame( uint32 pcount, ext_func_frame_t frameEndFunc = 0 )
   {
      StackFrame* frame = createFrame( pcount, frameEndFunc );
      addFrame( frame );
      return frame;
   }

   /** Adds a prepared stack frame on top of the frame list.
    *
    */
   void addFrame( StackFrame* frame );

   /** Removes the topmost frame.
    * The frame can be deleted or disposed via disposeFrame for further recycling.
    * \return The just removed topmost frame or 0 if the frame stack is empty.
    */
   StackFrame* popFrame();

   void pushTry( uint32 locationPC );
   void popTry( bool bMoveTo );

   /** Returns from the current stack frame.
    Modifies context PC and PCNext, and reutrns true if this is a break context.
    */
   bool callReturn();


   bool atomicMode() const { return m_atomicMode; }
   void atomicMode( bool b ) { m_atomicMode = b; }

   /** Adds some space in the local stack for local variables. */
   void addLocals( uint32 space )
   {
      if ( stack().length() < space )
         stack().resize( space );
   }

   /** Returns the nth local item.
      The first variable in the local context is numbered 0.
      \note Fetched item pointers are valid while the stack doesn't change.
            Pushes, addLocal(), item calls and VM operations may alter the
            stack. Using this method again after such operations allows to
            get a valid pointer to the desired item again. Items extracted with
            this method can be also saved locally in an Item instance, at
            the cost of a a flat item copy (a few bytes).
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   const Item *local( uint32 itemId ) const
   {
      return stack()[ itemId ].dereference();
   }

   /** Returns the nth local item.
      This is just the non-const version.
      The first variable in the local context is numbered 0.
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   Item *local( uint32 itemId )
   {
      return stack()[ itemId ].dereference();
   }

   /** Returns the parameter count for the current function.
      \note If the method as not a current frame, you'll have a crash.
      \return parameter count for the current function.
   */
   int32 paramCount() const {
      fassert( m_frames != 0 );
      return m_frames->m_param_count;
   }

   /** Returns the nth paramter passed to the VM.
      Const version of param(uint32).
   */
   const Item *param( uint32 itemId ) const
   {
      if ( itemId >= m_frames->m_param_count ) return 0;
      return m_frames->m_params[ itemId ].dereference();
   }

   /** Returns the nth paramter passed to the VM.

      This is just the noncost version.

      The count is 0 based (0 is the first parameter).
      If the parameter exists, a pointer to the Item holding the
      parameter will be returned. If the item is a reference,
      the referenced item is returned instead (i.e. the parameter
      is dereferenced before the return).

      The pointer may be modified by the caller, but this will usually
      have no effect in the calling program unless the parameter has been
      passed by reference.

      \note Fetched item pointers are valid while the stack doesn't change.
            Pushes, addLocal(), item calls and VM operations may alter the
            stack. Using this method again after such operations allows to
            get a valid pointer to the desired item. Items extracted with
            this method can be also saved locally in an Item instance, at
            the cost of a a flat item copy (a few bytes).

      \param itemId the number of the parameter accessed, 0 based.
      \return a valid pointer to the (dereferenced) parameter or 0 if itemId is invalid.
      \see isParamByRef
   */
   Item *param( uint32 itemId )
   {
      if ( itemId >= m_frames->m_param_count ) return 0;
      //return &m_frames->prev()->stack()[ m_frames->prev()->stack().length() - m_frames->m_param_count + itemId];
      return m_frames->m_params[ itemId ].dereference();
   }

   /** Returns the nth pre-paramter passed to the VM.
      Pre-parameters can be used to pass items to external functions.
      They are numbered 0...n in reverse push order, and start at the first
      push before the first real parameter.

      For example;

      \code
         Item *p0, *p1, *callable;
         ...
         vm->pushParameter( (int64) 0 );   // pre-parameter 1
         vm->pushParameter( vm->self() );  // pre-parameter 0
         vm->pushParameter( *p0 );
         vm->pushParameter( *p1 );

         vm->callFrame( *callable, 2 );    // 2 parameters
      \endcode
   */
   Item *preParam( uint32 itemId )
   {
      Item* params = m_frames->m_params;
      params -= 1 + itemId;
      return params->dereference();
   }

   /** Const version of preParam */
   const Item *preParam( uint32 itemId ) const
   {
      Item* params = m_frames->m_params;
      params -= 1 + itemId;
      return params->dereference();
   }

   /** Returns true if the nth element of the current function has been passed by reference.
      \param itemId the number of the parameter accessed, 0 based.
      \return true if the parameter exists and has been passed by reference, false otherwise
   */
   bool isParamByRef( uint32 itemId ) const
   {
      if ( itemId >= m_frames->m_param_count ) return 0;
      return m_frames->m_params[ itemId ].type() == FLC_ITEM_REFERENCE;
   }

   /** Installs a post-processing return frame handler.
      The function passed as a parmeter will receive a pointer to this VM.

      The function <b>MUST</b> return true if it performs another frame item call. This will
      tell the VM that the stack cannot be freed now, as a new call stack has been
      prepared for immediate execution. When done, the function will be called again.

      A frame handler willing to call another frame and not willing to be called anymore
      must first unininstall itself by calling this method with parameters set at 0,
      and then it <b>MUST return true</b>.

      A frame handler not installing a new call frame <b>MUST return false</b>. This will
      terminate the current stack frame and cause the VM to complete the return stack.
      \param callbackFunct the return frame handler, or 0 to disinstall a previously set handler.
   */

   void returnHandler( ext_func_frame_t callbackFunc )
   {
      currentFrame()->m_endFrameFunc = callbackFunc;
   }

   ext_func_frame_t returnHandler() const
   {
      return currentFrame()->m_endFrameFunc;
   }

   /** Pushes a parameter for the vm callItem and callFrame functions.
      \see callItem
      \see callFrame
      \param item the item to be passes as a parameter to the next call.
   */
   void pushParam( const Item &item ) { stack().append(item); }

   StackFrame* allocFrame();
   void disposeFrame( StackFrame* frame );
   void disposeFrames( StackFrame* first, StackFrame* last );
   void resetFrames();

   void fillErrorTraceback( Error& err );
   
   //! Gets a step in a traceback.
   bool getTraceStep( uint32 level, const Symbol* &sym, uint32& line, uint32 &pc );
};

}

#endif

/* end of flc_vmcontext.h */
