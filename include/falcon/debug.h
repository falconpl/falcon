/*
   FALCON - The Falcon Programming Language.
   FILE: debug.h

   Falcon debugging system.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Thur, 26 Nov 2009 06:08:00 +1200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/genericmap.h>
#include <falcon/genericlist.h>
#include <falcon/symbol.h>

namespace Falcon
{

   class FALCON_DYN_CLASS DebugVMachine : public VMachine
   {

   public:
      DebugVMachine();

      
      void setBreakPoint(Symbol* func, int32 lineNo);

      void breakPoint();
      void resume();
      void stop();
      void step();
      void runTo();
      void restart();
      void stepInto();
      void stepOut();

      void setWatch(String name);
      void removeWatch(String name);
      const Map& watches() const;

      const ItemArray& stackTrace() const;



   private:

      //map to hold breakpoints
      Map m_breakPoints;

      Map m_watches;

      bool m_step;
      bool m_stepInto;
      bool m_stepOut;

      virtual void periodicCallback();


   };

} //end namespace Falcon
