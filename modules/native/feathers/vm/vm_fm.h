/* FALCON - The Falcon Programming Language.
 * FILE: vm_fm.h
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

#ifndef FALCON_FEATHERS_VM_FM_H
#define FALCON_FEATHERS_VM_FM_H

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

//Define the math_extra module class
class ModuleVM: public Falcon::Module
{
public:
   ModuleVM();
   virtual ~ModuleVM();
};

}}

#endif

/* end of vm_fm.h */

