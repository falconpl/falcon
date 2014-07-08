/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_fm.h

   Configuration parser module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 08 Aug 2013 18:28:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_MODULE_CONFPARSER
#define FALCON_FEATHERS_MODULE_CONFPARSER

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleConfparser: public Module
{
public:
   ModuleConfparser();
   virtual ~ModuleConfparser();
};

}
}

#endif

/* end of confparser_fm.h */
