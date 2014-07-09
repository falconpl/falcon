/* FALCON - The Falcon Programming Language.
 * FILE: sys_fm.h
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

#ifndef FALCON_FEATHERS_SYS_FM_H
#define FALCON_FEATHERS_SYS_FM_H

#define FALCON_FEATHER_SYS_NAME "sys"

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

//Define the math_extra module class
class ModuleSys: public Falcon::Module
{
public:
   // initialize the module
   ModuleSys();
   virtual ~ModuleSys();
};

}}
#endif

/* end of sys_fm.h */

