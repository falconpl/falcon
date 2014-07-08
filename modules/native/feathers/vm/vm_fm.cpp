/* FALCON - The Falcon Programming Language.
 * FILE: vm_fm.cpp
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

#include "vm_ext.h"
#include "vm_fm.h"

#include "version.h"

namespace Falcon {
namespace Feathers {


ModuleVM::ModuleVM():
   Module("vm")
{
   addObject(new Falcon::Ext::ClassVM, true);
}

ModuleVM::~ModuleVM() {}

}
}

/* end of vm_fm.cpp */

