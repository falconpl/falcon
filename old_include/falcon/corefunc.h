/*
   FALCON - The Falcon Programming Language.
   FILE: corefunc.h

   Language level live function object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Jan 2009 14:54:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Abstract live function object.
*/

#ifndef FLC_CORE_FUNC_H
#define FLC_CORE_FUNC_H

#include <falcon/symbol.h>
#include <falcon/livemodule.h>

#include <falcon/callpoint.h>

namespace Falcon
{

class LiveModule;
class VMachine;
class ItemArray;

/** Class implementing a live function in the VM.
*/
class FALCON_DYN_CLASS CoreFunc: public CallPoint
{
   LiveModule *m_lm;
   const Symbol* m_symbol;
   ItemArray* m_closure;

public:

   /** Creates a Falcon function.
      The symbol determines if this will be an external or internal function.
   */
   CoreFunc( const Symbol *sym, LiveModule *lm ):
      m_lm( lm ),
      m_symbol( sym ),
      m_closure(0)
   {}

   CoreFunc( const CoreFunc& other ):
      m_closure(0)
   {
      m_lm = other.m_lm;
      m_symbol = other.m_symbol;
   }

   virtual ~CoreFunc();

   LiveModule *liveModule() const { return m_lm; }
   const Symbol *symbol() const { return m_symbol; }
   ItemArray* closure() const  { return m_closure; }
   void closure( ItemArray* cl ) { m_closure = cl; }

   virtual void readyFrame( VMachine* vm, uint32 paramCount );
   virtual bool isFunc() const { return true; }
   virtual const String& name() const { return m_symbol->name(); }

   virtual void gcMark( uint32 gen );
};

}

#endif
/* end of corefunc.h */
