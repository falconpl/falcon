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
   void parseCommand( const String& line, VMContext* ctx );

   void printCode( VMContext* ctx );
   void printLoc( VMContext* ctx );

   void displayStack( VMContext* ctx, int64 depth );
   void displayDyns( VMContext* ctx, int64 depth );
   void displayCode( VMContext* ctx, int64 depth );
   void displayCall( VMContext* ctx, int64 depth );
   void displayBack( VMContext* ctx, int64 depth );
   void displayGlobals( VMContext* ctx );

   void exitDebugger() { m_bActive = false; }
   void listCommands() const;
   void describe(const String& cmd) const;
   void write(const String& str) const;
   void writeLine(const String& str) const;

   void showCode() { m_showCode = true; }

   PStep* m_stepPostEval;
   PStep* m_stepAfterNext;

   TextWriter* writer() const { return m_tw; }

private:
   class PStepPostEval;
   class PStepAfterNext;
   class PStepCatcher;

   String m_lastCommand;
   bool m_hello;
   bool m_showCode;
   bool m_bActive;

   TextWriter* m_tw;

   class Private;
   Private* _p;
};

}
#endif

/* end of debugger.h */
