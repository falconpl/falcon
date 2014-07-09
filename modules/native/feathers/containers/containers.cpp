/* FALCON - The Falcon Programming Language.
 * FILE: containers.cpp
 * 
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 * 
 * See LICENSE file for licensing details.
 */

#include "containers_fm.h"

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::Feathers::ModuleContainers;
   return mod;
}

#endif

/* end of containers.cpp */
