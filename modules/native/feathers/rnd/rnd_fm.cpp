/* FALCON - The Falcon Programming Language.
 * FILE: rnd_fm.cpp
 * 
 * Extra functions
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 24 Feb 2013 22:37:06 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: FALCON AUTHORS
 * 
 * Licensed under the Falcon Programming Language License,
 * Version 1.1 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain
 * a copy of the License at
 * 
 * http://www.falconpl.org/?page_id=license_1_1
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#undef SRC
#define SRC "modules/native/feathers/rnd/rnd_fm.cpp"

#include "rnd_fm.h"
#include "rnd_ext.h"

namespace Falcon { 
namespace Feathers {

//Define the math_extra module class

// initialize the module
ModuleRnd::ModuleRnd():
   Module("rnd", true)
{
   // Standard
   *this
      << new Falcon::Ext::FALCON_FUNCTION_NAME(random)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomChoice)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomPick)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomWalk)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomGrab)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomDice)
      << new Falcon::Ext::FALCON_FUNCTION_NAME(randomSeed)
            ;

}
ModuleRnd::~ModuleRnd() {}

}
} // namespace Falcon::Ext

/* end of rnd_fm.cpp */

