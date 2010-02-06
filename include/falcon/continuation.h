/*
   FALCON - The Falcon Programming Language.
   FILE: continuation.h

   Continuation object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Dec 2009 17:04:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CONTINUATION_H_
#define FALCON_CONTINUATION_H_

#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>

namespace Falcon
{

class Continuation: public BaseAlloc
{
public:
   /** Form the continuation.
       The place where the continuation is formed will be used
       as the return point when the invoke method is called.
    */
   Continuation( VMachine* vm );

   Continuation( const Continuation& e );
   virtual ~Continuation();

   /** Applies the continuation on the virtual machine.
       If the continuation is
    */
   bool jump();

   /** Invoke the continuation.
    The effect of the invocation is that of:
    1) saving the current context.
    2) returning the given value to the original caller.
    3) applying the original context and unrolling the stack up to the call level.
    */
   void suspend( const Item& retval );

   void reset()
   {
      m_bComplete = false;
      m_tgtSymbol = 0;
   }

   /** Returns true if the topmost code in this continuation has been fully executed.

       In other words, it returns false if it has been not yet executed or
       executed but suspended.
    * */
   bool complete() const {
      return m_tgtSymbol != 0 && m_tgtPC >= m_tgtSymbol->getFuncDef()->codeSize();
   }

   bool ready() const {
      return m_tgtSymbol == 0;
   }

   ItemArray &params() { return m_params; }
   StackFrame* frames() const { return m_top; }

   // the first parameter of the bottom call...
   bool updateSuspendItem( const Item& itm );

private:

   VMachine* m_vm;

   VMContext* m_context;
   /** Level at which the stack is created */
   uint32 m_stackLevel;

   /** Symbol/module where the continuation is invoked. */
   const Symbol* m_tgtSymbol;

   /** Symbol/module where the continuation is invoked. */
   LiveModule *m_tgtLModule;

   uint32 m_tgtPC;

   /** Frame where the call is started. */
   StackFrame* m_callingFrame;

   /** Current frame when the continuation is called. */
   StackFrame* m_top;

   /** First frame on top of the calling frame. */
   StackFrame* m_bottom;

   /** Parameters for the bottom frame (safely stored here) */
   ItemArray m_params;

   bool m_bComplete;

   int32 m_refCount;
};


class ContinuationCarrier: public CoreObject
{
public:
   ContinuationCarrier( const CoreClass* cc );
   ContinuationCarrier( const ContinuationCarrier& other );

   virtual ~ContinuationCarrier();
   virtual ContinuationCarrier *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &prop, Item &value ) const;

   Continuation* cont() const { return m_cont; }
   void cont( Continuation* cc ) { m_cont = cc; }
   virtual void gcMark( uint32 mark );

   const Item& ccItem() const { return m_citem; }
   void ccItem( const Item& itm ) { m_citem = itm; }
   const Item& suspendItem() const  { return m_suspendItem; }
   Item& suspendItem()  { return m_suspendItem; }

   static CoreObject* factory( const CoreClass* cls, void* data, bool deser );
private:
   Continuation* m_cont;
   Item m_citem;
   Item m_suspendItem;
   uint32 m_mark;
};

}

#endif /* CONTINUATION_H_ */
