/* FALCON - The Falcon Programming Language.
 * FILE: containers_fm.cpp
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

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "containers_fm.h"
#include "version.h"

#include "classcontainer.h"
#include "classlist.h"
#include "classiterator.h"
#include "errors.h"

/*#
   @module feathers.containers
   @brief Interface to the Virtual Machine running this process.
*/

namespace Falcon {

ModuleContainers::ModuleContainers():
      Module("containers")
{
   m_containerClass = new ClassContainer;
   m_iteratorClass = new ClassIterator;

   *this
      << m_containerClass
      << m_iteratorClass
      << new ClassContainerError
      << new ClassList( m_containerClass );
}

ModuleContainers::~ModuleContainers()
{
}

}

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::ModuleContainers;
   return mod;
}

/* end of containers.cpp */

