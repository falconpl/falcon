/*
   FALCON - The Falcon Programming Language.
   FILE: breakcallback.h

   Falcon virtual machine -- callback routine invoked at breakpoints
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_BREAKCALLBACK_H_
#define _FALCON_BREAKCALLBACK_H_

#include <falcon/setup.h>

namespace Falcon {

class VMContext;
class Processor;
class Process;

/** Callback routine invoked at breakpoints

   This class is a base functor used as interface for callbacks on breakpoints.

   When the processor finds a breakpoint event, it invokes this routine in the host
   process, provided it was already installed, or eventually exits.

   The instance of this class is then able to inspect the context, modify it and
   eventually set new events.
 */
class FALCON_DYN_CLASS BreakCallback
{
public:

   virtual ~BreakCallback() {}

   virtual void onInstalled( Process* ) {}
   virtual void onUnistalled( Process* ) {}

   virtual void onBreak( Process* p, Processor* pr, VMContext* ctx ) = 0;

};

}
#endif

/* end of breakcallback.h */
