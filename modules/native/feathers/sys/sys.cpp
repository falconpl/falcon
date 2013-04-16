/* FALCON - The Falcon Programming Language.
 * FILE: sys.cpp
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
#include "sys_ext.h"

#include "version.h"

/*#
   @module feathers.sys
   @brief Interface to system services.

   - Environment variables
   - process standard stream (overriding virual machine ones).
   - host process control
*/

//Define the math_extra module class
class SysModule: public Falcon::Module
{
public:
   // initialize the module
   SysModule():
      Module("sys")
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
   virtual ~SysModule() {}
};

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new SysModule;
   return mod;
}

/* end of sys.cpp */

