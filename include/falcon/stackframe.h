/*
   FALCON - The Falcon Programming Language.
   FILE: stackframe.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#ifndef flc_stackframe_H
#define flc_stackframe_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/basealloc.h>
#include <falcon/itemarray.h>

namespace Falcon {

class Module;

class StackFrame: public BaseAlloc
{
public:
   enum
   {
      pa_default = 8
   } constants;

   bool m_break;

   uint32 m_ret_pc;
   uint32 m_call_pc;
   uint32 m_param_count;
   uint32 m_try_base;

   const Symbol *m_symbol;
   LiveModule *m_module;
   ext_func_frame_t m_endFrameFunc;
   StackFrame* m_prevTryFrame;

   Item m_self;
   Item m_binding;

   // points to the parameter part in the previous area.
   Item* m_params;

   StackFrame( int preAlloc = pa_default ):
      m_symbol(0),
      m_module(0),
      m_prevTryFrame(0),
      m_prev(0),
      m_stack( preAlloc )
   {}

   StackFrame( const StackFrame& other );

   /** Retrieves the previous frame in the call stack. */
   StackFrame* prev() const { return m_prev; }
   void prev( StackFrame* p ) { m_prev = p; }

   /** Returns the items in the stack. */
   const ItemArray& stack() const { return m_stack; }
   ItemArray& stack() { return m_stack; }

   /** Remvoves N elements from thes stack */
   void pop( uint32 size )  { m_stack.resize( m_stack.length() - size ); }

   /** Allocates new space in the stack. */
   void push( uint32 size ) { m_stack.resize( m_stack.length() + size ); }

   Item* stackItems() const { return m_stack.elements(); }

   uint32 stackSize() const { return m_stack.length(); }

   void pushItem( const Item& v ) { m_stack.append(v); }
   const Item& topItem() const { return m_stack.back(); }
   Item& topItem() { return m_stack[m_stack.length()-1]; }

   void popItem( Item &tgt ) { tgt = m_stack.back(); m_stack.resize( m_stack.length() - 1 ); }

   void resizeStack( uint32 size ) { m_stack.resize( size ); }

   void prepareParams( StackFrame* previous, uint32 paramCount )
   {
      if( paramCount != 0 )
      {
         m_params = previous->m_stack.elements() + previous->m_stack.length() - paramCount;
      }
      else
         m_params = 0;
   }

   const Item& localItem( uint32 id ) const { return m_stack[id]; }
   Item& localItem( uint32 id ) { return m_stack[id]; }

   /** Copy a whole hierarcy of frames. */
   StackFrame* copyDeep( StackFrame** bottom );

   void gcMark( uint32 mark );

private:
   StackFrame* m_prev;
   ItemArray m_stack;
} ;


}

#endif

/* end of stackframe.h */
