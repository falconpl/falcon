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

#include "sys_fm.h"

/*#
   @module sys
   @ingroup feathers
   @brief Interface to system services.

   - Environment variables
   - process standard stream (overriding virual machine ones).
   - host process control
*/

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::Feathers::ModuleSys;
   return mod;
}

#endif

/* end of sys.cpp */

