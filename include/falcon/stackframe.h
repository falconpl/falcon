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
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Module;
class ItemVector;

typedef struct tag_StackFrame
{
   Item header;

   const Symbol *m_symbol;
   uint32 m_ret_pc;
   uint32 m_call_pc;
   LiveModule *m_module;
   bool m_break;
   uint16 m_param_count;
   uint32 m_stack_base;
   uint32 m_try_base;
   Item m_self;
   Item m_binding;

   ext_func_frame_t m_endFrameFunc;
} StackFrame;

void StackFrame_deletor( void *data );

class StackFrameList: public List
{
   friend void StackFrame_deletor( void *data );

public:
   StackFrameList():
      List( StackFrame_deletor )
   {}
};


#define VM_FRAME_SPACE (sizeof(StackFrame) / sizeof(Item) + 1)

}

#endif

/* end of stackframe.h */
