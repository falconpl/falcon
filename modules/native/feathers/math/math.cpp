/* FALCON - The Falcon Programming Language.
 * FILE: math_extra.cpp
 * 
 * Extra math functions
 * Main module file, providing the module object to the Falcon engine.
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
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "math_fm.h"

/*#
   @module math
   @ingroup feathers
   @brief Mathematical functions

   The @b math module provides standard mathematical function that
   operate on IEEE 64 bit double-precision floating point numbers.
   
   Functions available in this module cover:
   - Trigonometry;
   - Logarithms and exponentials;
   - Floating point manipulation;
   - Combinatory calculus.
*/

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::Feathers::ModuleMath;
   return mod;
}

/* end of math_extra.cpp */

