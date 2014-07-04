/* FALCON - The Falcon Programming Language.
 * FILE: vm.cpp
 * 
 * Interface to the virtual machine
 * Main module file, providing the module object to the Falcon engine.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Wed, 06 Mar 2013 17:24:56 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: The above AUTHOR
 * 
 * See LICENSE file for licensing details.
 */

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "vm_fm.h"

/*#
   @module vm
   @ingroup feathers
   @brief Interface to the Virtual Machine running this process.
*/


FALCON_MODULE_DECL
{
   Falcon::Module* mod = new ::Falcon::Feathers::ModuleVM;
   return mod;
}

/* end of vm.cpp */

