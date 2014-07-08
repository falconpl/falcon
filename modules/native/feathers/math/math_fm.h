/* FALCON - The Falcon Programming Language.
 * FILE: math_ext.h
 * 
 * Extra math functions
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Steven N Oliver
 * Begin: Wed, 27 Oct 2010 20:12:51 -0400
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2010: The above AUTHOR
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

/** \file
  Extra math functions
  Interface extension functions - header file
  */

#ifndef FALCON_FEATHERS_MATH_FM_H
#define FALCON_FEATHERS_MATH_FM_H

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleMath: public Module
{
public:
   ModuleMath();
   virtual ~ModuleMath();
};

}
}

#endif

/* end of math_ext.h */

