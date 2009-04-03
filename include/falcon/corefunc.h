/*
   FALCON - The Falcon Programming Language.
   FILE: corefunc.h

   Abstract live function object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Jan 2009 14:54:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Abstract live function object.
*/

#ifndef FLC_CORE_FUNC_H
#define FLC_CORE_FUNC_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/symbol.h>
#include <falcon/livemodule.h>

namespace Falcon
{

class LiveModule;
class VMachine;

/** Class implementing a live function in the VM.
*/
class FALCON_DYN_CLASS CoreFunc: public Garbageable
{
   LiveModule *m_lm;
   const Symbol* m_symbol;

public:

   /** Creates a Falcon function.
      The symbol determines if this will be an external or internal function.
   */
   CoreFunc( const Symbol *sym, LiveModule *lm ):
      Garbageable(),
      m_lm( lm ),
      m_symbol( sym )
   {}

   CoreFunc( const CoreFunc& other ):
      Garbageable()
   {
      m_lm = other.m_lm;
      m_symbol = other.m_symbol;
   }

   virtual ~CoreFunc() {}

   LiveModule *liveModule() const { return m_lm; }
   const Symbol *symbol() const { return m_symbol; }
   const String& name() const { return m_symbol->name(); }

   bool isValid() const { return m_lm->isAlive(); }

   void readyFrame( VMachine *vm, uint32 params );
};

}

#endif
/* end of corefunc.h */
