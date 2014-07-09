/*
   FALCON - The Falcon Programming Language.
   FILE: shmem_fm.cpp

   Shared memory mapped object.

   Interprocess shared-memory object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Nov 2013 13:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "shmem_fm.h"

/*#
   @module shmem
   @ingroup feathers
   @brief Shared and persistent memory extsnsions.
*/

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::Feathers::ModuleShmem;
   return mod;
}

#endif
/* end of shmem_fm.cpp */

