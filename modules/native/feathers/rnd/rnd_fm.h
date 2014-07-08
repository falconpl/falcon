/* FALCON - The Falcon Programming Language.
 * FILE: rnd_fm.h
 * 
 * Simple random functions
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 24 Feb 2013 22:37:06 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: The above AUTHOR
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

#ifndef FALCON_FEATHERS_RND_FM_H
#define FALCON_FEATHERS_RND_FM_H

#include <falcon/module.h>

namespace Falcon { 
namespace Feathers {

//Define the math_extra module class
class ModuleRnd: public Falcon::Module
{
public:
 // initialize the module
 ModuleRnd();
 virtual ~ModuleRnd();
};
}
} // namespace Falcon::Ext

#endif

/* end of rnd_fm.h */

