/*
   FALCON - The Falcon Programming Language.
   FILE: debugger.h

   Falcon virtual machine -- Basic standard debugger-in-a-box
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DEBUGGER_H_
#define _FALCON_DEBUGGER_H_

#include <falcon/setup.h>
#include <falcon/breakcallback.h>
#include <falcon/string.h>

namespace Falcon {

class VMContext;
class Processor;
class TextWriter;
class PStep;

/** Basic standard debugger-in-a-box.

   This class is a minimally functional debugger that plugs in the
   break-callback feature of VM processors to give.
 */
class FALCON_DYN_CLASS Debugger: public BreakCallback
{
public:
   Debugger();
   virtual ~Debugger();

   virtual void onBreak( Process* p, Processor* pr, VMContext* ctx );

   bool parseCommand( TextWriter& wr, const String& line, VMContext* ctx );
   void printCode( TextWriter& wr, VMContext* ctx );
   void printLoc( TextWriter& wr, VMContext* ctx );

   void displayStack( TextWriter& wr, VMContext* ctx, int64 depth );
   void displayDyns( TextWriter& wr, VMContext* ctx, int64 depth );
   void displayCode( TextWriter& wr, VMContext* ctx, int64 depth );
   void displayCall( TextWriter& wr, VMContext* ctx, int64 depth );
   void displayGlobals( TextWriter& wr, VMContext* ctx );

private:
   PStep* m_stepPostEval;
   class PStepPostEval;

   PStep* m_stepAfterNext;
   class PStepAfterNext;

   class PStepCatcher;

   String m_lastCommand;
   bool m_hello;


};

}
#endif

/* end of debugger.h */
