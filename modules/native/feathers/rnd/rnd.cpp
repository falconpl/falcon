/* FALCON - The Falcon Programming Language.
 * FILE: rnd.cpp
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

#include "rnd_fm.h"

/*#
   @module rnd
   @ingroup feathers
   @brief Simple pseudo-random number generator functions.

   This module exposes a set of simple function-oriented
   pseudo-random generator function that use the Mersenne-Twister
   interlocked pseudo-random generator available in the Falcon
   Virtual Machine.

   This means that the status of the number generator is shared
   across all the processes and contexts that participate in a
   virtual machine.

   In case the application needs a fixed seed to generate a
   repeatable random sequence, this set of function can be considered
   consistent in full stand-alone applications only.

   The VM pseudo-random number generator is also used by third party
   modules to give the application a simple mean to configure the
   number generator.
   
   @note The @b rand module provides a more general support to
   pseudo-random number generation.
*/


FALCON_MODULE_DECL
{
   Falcon::Module* mod = new ::Falcon::Feathers::ModuleRnd;
   return mod;
}

/* end of rnd.cpp */

