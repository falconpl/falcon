/*
   FALCON - The Falcon Programming Language.
   FILE: flc_vmcontext.h
   $Id: vmcontext.h,v 1.8 2007/08/18 11:08:06 jonnymind Exp $

   Virtual Machine coroutine execution context.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

namespace Falcon {

class VMachine;
class Symbol;
class Item;
class ItemVector;

/** Class representing a coroutine execution context. */
class FALCON_DYN_CLASS VMContext: public BaseAlloc
{
   Symbol* m_symbol;
   uint32 m_moduleId;

   Item m_regA;
   Item m_regB;
   Item m_regS1;
   Item m_regS2;

   ItemVector *m_stack;
   uint32 m_stackBase;
   byte *m_code;
   uint32 m_pc;
   uint32 m_pc_next;

   numeric m_schedule;
   int32 m_priority;

   uint32 m_tryFrame;

public:
   VMContext( VMachine *origin );
   ~VMContext();

   void save( const VMachine *origin );
   void restore( VMachine *target )  const;

   void priority( int32 value ) { m_priority = value; }
   int32 priority() const { return m_priority; }

   void schedule( numeric value ) { m_schedule = value; }
   numeric schedule() const { return m_schedule; }

   Item &regA() { return m_regA; }
   const Item &regA() const { return m_regA; }
   Item &regB() { return m_regB; }
   const Item &regB() const { return m_regB; }

   Item &self() { return m_regS1; }
   const Item &self() const { return m_regS1; }
   Item &sender() { return m_regS2; }
   const Item &sender() const { return m_regS2; }

   void stackBase( uint32 pos ) { m_stackBase = pos; }
   uint32 stackBase() const { return m_stackBase; }

   ItemVector *getStack() const { return m_stack; }
};

}

#endif

/* end of flc_vmcontext.h */
