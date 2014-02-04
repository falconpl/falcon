/* FALCON - The Falcon Programming Language.
 * FILE: containers_fm.h
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

#ifndef _FALCON_FEATHERS_CONTAINERS_FM_H_
#define _FALCON_FEATHERS_CONTAINERS_FM_H_

#include <falcon/module.h>

namespace Falcon {

// Module
class ModuleContainers: public Module
{
public:
   // initialize the module
   ModuleContainers();
   virtual ~ModuleContainers();

   const Class* iteratorClass() const { return m_iteratorClass; }
   const Class* containerClass() const { return m_containerClass; }

public:
   Class* m_iteratorClass;
   Class* m_containerClass;
};

}

#endif

/* end of containers_fm.h */

