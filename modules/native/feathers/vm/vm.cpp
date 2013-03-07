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

#include <falcon/module.h>
#include "vm_ext.h"

#include "version.h"

/*#
   @module feathers.vm
   @brief Interface to the Virtual Machine running this process.
*/

//Define the math_extra module class
class VMModule: public Falcon::Module
{
public:
   // initialize the module
   VMModule():
      Module("vm")
   {
      addObject(new Falcon::Ext::ClassVM, true);
   }
   virtual ~VMModule() {}
};

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new VMModule;
   return mod;
}

/* end of vm.cpp */

