/*
   FALCON - The Falcon Programming Language.
   FILE: debug.cpp

   Falcon debugging system.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Thur, 26 Nov 2009 06:08:00 +1200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#include <falcon/debug.h>

namespace Falcon
{

   DebugVMachine::DebugVMachine() : 
      VMachine(), m_breakPoints(Map(&traits::t_voidp(),&traits::t_List())), m_watches(Map(&traits::t_string(),&traits::t_voidp())),
      m_step(false), m_stepInto(false), m_stepOut(false)
   {
      callbackLoops(1);
      
   }

   const Map& DebugVMachine::watches() const
   {
      return m_watches;
   }

   const ItemArray& DebugVMachine::stackTrace() const
   {
      return stack();
   }

   void DebugVMachine::periodicCallback()
   {
      m_opNextCheck = m_opCount + 1;
   }

} //end namespace Falcon
