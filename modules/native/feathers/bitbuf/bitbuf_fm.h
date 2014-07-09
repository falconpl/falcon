/*
   FALCON - The Falcon Programming Language.
   FILE: module_bitbuf.h

   Buffering extensions
   Main module entity
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Jul 2013 13:22:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

   http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

#ifndef FALCON_FEATHERS_MODULE_BITBUF_H
#define FALCON_FEATHERS_MODULE_BITBUF_H

#define FALCON_FEATHER_BITBUF_NAME "bitbuf"

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleBitbuf:public Module
{
public:
   ModuleBitbuf();
   virtual ~ModuleBitbuf();
};

}
}

#endif

/* end of module_bitbuf.h */
