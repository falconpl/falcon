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

#ifndef flc_flc_vmcontext_H
#define flc_flc_vmcontext_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/genericvector.h>
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>
#include <falcon/livemodule.h>

namespace Falcon {

class Symbol;
class Item;
class ItemVector;
class VMSemaphore;

/** Class representing a coroutine execution context. */
class FALCON_DYN_CLASS VMContext: public BaseAlloc
{
   Item m_regA;
   Item m_regB;

   Item m_regS1;
   Item m_regL1;
   Item m_regL2;
   Item m_regBind;
   Item m_regBindP;

   ItemVector *m_stack;
   VMSemaphore *m_sleepingOn;

   /** Currently executed symbol.
      May be 0 if the startmodule has not a "__main__" symbol;
      this should be impossible when things are set up properly
      \todo, make always nonzeor
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
      changing this value, that is normally set just to m_pc + the lenght
      of the current instruction.
   */
   uint32 m_pc_next;

   /** Stack base is the position of the current stack frame.
      As there can't be any stack frame at 0, a position of 0 means that the VM is running
      the global module code.
   */
   uint32 m_stackBase;

   /** Position of the topmost try frame handler. */
   uint32 m_tryFrame;

   friend class VMSemaphore;

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

   uint32& stackBase() { return m_stackBase; }
   const uint32& stackBase() const { return m_stackBase; }

   uint32& tryFrame() { return m_tryFrame; }
   const uint32& tryFrame() const { return m_tryFrame; }

   Item &regA() { return m_regA; }
   const Item &regA() const { return m_regA; }
   Item &regB() { return m_regB; }
   const Item &regB() const { return m_regB; }
   Item &regBind() { return m_regBind; }
   const Item &regBind() const { return m_regBind; }
   Item &regBindP() { return m_regBindP; }
   const Item &regBindP() const { return m_regBindP; }

   Item &self() { return m_regS1; }
   const Item &self() const { return m_regS1; }

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

   ItemVector &stack() { return *m_stack; }
   const ItemVector &stack() const { return *m_stack; }

   VMSemaphore *sleepingOn() const { return m_sleepingOn; }
   void sleepOn( VMSemaphore *sl ) { m_sleepingOn = sl; }

   /** Returns the current module global variables vector. */
   ItemVector &globals() { return m_lmodule->globals(); }

   /** Returns the current module global variables vector (const version). */
   const ItemVector &globals() const { return m_lmodule->globals(); }

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
      fassert( m_symbol->isFunction() );
      return m_symbol->getFuncDef()->code();
   }
};

}

#endif

/* end of flc_vmcontext.h */
