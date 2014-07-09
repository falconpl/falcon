/*
   FALCON - The Falcon Programming Language.
   FILE: modcmdlineparser.h

   Command line parser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 08 Aug 2013 18:28:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_MODULE_CMDLINE
#define FALCON_FEATHERS_MODULE_CMDLINE

#define FALCON_FEATHER_CMDLINE_NAME "cmdline"

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon {

class PStep;

namespace Feathers {

class ModuleCmdline: public Module
{
public:
   // step called to get the next option
   PStep* m_stepGetOption;
   // step landing after a callback
   PStep* m_stepAfterCall;

   ModuleCmdline();
   virtual ~ModuleCmdline();
};

}
}

#endif

/* end of modcmdlineparser.h */
