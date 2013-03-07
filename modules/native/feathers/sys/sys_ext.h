/* FALCON - The Falcon Programming Language.
 * FILE: vm_ext.h
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

#ifndef FALCON_EXT_SYS_H
#define FALCON_EXT_SYS_H

#include <falcon/module.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/class.h>

namespace Falcon { 
    namespace Ext {
       FALCON_DECLARE_FUNCTION(stdIn,"");
       FALCON_DECLARE_FUNCTION(stdOut,"");
       FALCON_DECLARE_FUNCTION(stdErr,"");
       FALCON_DECLARE_FUNCTION(getEnv,"variable:S");
       FALCON_DECLARE_FUNCTION(setEnv,"variable:S,value:S");
       FALCON_DECLARE_FUNCTION(environ,"");
       FALCON_DECLARE_FUNCTION(edesc,"code:N");

       FALCON_DECLARE_FUNCTION(cores,"");
       FALCON_DECLARE_FUNCTION(epoch,"");
       FALCON_DECLARE_FUNCTION(systemType,"");

       // todo UID, PID, GID etc.
    }
} // namespace Falcon::Ext

#endif

/* end of vm_ext.h */

