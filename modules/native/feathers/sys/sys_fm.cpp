/* FALCON - The Falcon Programming Language.
 * FILE: sys_fm.cpp
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

#include "sys_fm.h"
#include "sys_ext.h"
#include "version.h"

namespace Falcon {
namespace Feathers {

// initialize the module
ModuleSys::ModuleSys():
   Module(FALCON_FEATHER_SYS_NAME)
{
    // As Windows and some UNIX flavor declare "environ" as a macro,
    // we are forced to give the function a special name, and re-define
    // its function-name in the module, after it's created.
   Falcon::Function *ef = new Falcon::Ext::Function_falcon_environ__;
   ef->name("environ");

   *this
    << new Falcon::Ext::Function_stdIn
    << new Falcon::Ext::Function_stdOut
    << new Falcon::Ext::Function_stdErr
    << new Falcon::Ext::Function_getEnv
    << new Falcon::Ext::Function_setEnv
    << ef
    << new Falcon::Ext::Function_edesc

    << new Falcon::Ext::Function_cores
    << new Falcon::Ext::Function_epoch
    << new Falcon::Ext::Function_systemType
             ;
}

ModuleSys::~ModuleSys()
{}

}}

/* end of sys_fm.cpp */

