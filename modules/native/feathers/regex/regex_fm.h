/*
   FALCON - The Falcon Programming Language.
   FILE: regex_fm.h

   Regular expression extension
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_REGEX_H
#define FALCON_FEATHERS_REGEX_H

#define FALCON_FEATHER_REGEX_NAME "regex"

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {


class ModuleRegex: public Falcon::Module
{
public:
   ModuleRegex();
   virtual ~ModuleRegex();
};

}
}
#endif

/* end of regex_fm.h */
