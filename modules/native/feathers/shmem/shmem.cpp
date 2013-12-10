/*
   FALCON - The Falcon Programming Language.
   FILE: shmem.cpp

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

#include <falcon/module.h>
#include "shmem_ext.h"
#include "session_ext.h"
#include "ipsem_ext.h"
#include "errors.h"

#include "version.h"

namespace Falcon {

/*#
   @module feathers.shmem
   @brief Shared and persistent memory extsnsions.

*/

class ShmemModule: public Falcon::Module
{
public:

   // initialize the module
   ShmemModule():
      Module("shmem")
   {
      *this
         << new ClassSharedMem
         << new ClassIPSem
         << new ClassShmemError
         << new ClassSession
         << new ClassSessionError
               ;
   }

   virtual ~ShmemModule() {}
};

}


FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::ShmemModule;
   return mod;
}

/* end of rnd.cpp */

