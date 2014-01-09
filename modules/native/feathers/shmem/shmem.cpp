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

#include "shmemmodule.h"
#include "shmem_ext.h"
#include "session_ext.h"
#include "session_srv.h"
#include "ipsem_ext.h"
#include "errors.h"

#include "version.h"

namespace Falcon {

/*#
   @module feathers.shmem
   @brief Shared and persistent memory extsnsions.
*/


// initialize the module
ShmemModule::ShmemModule():
   Module("shmem")
{
   m_classSession = new ClassSession;

   *this
      << new ClassSharedMem
      << new ClassIPSem
      << new ClassShmemError
      << m_classSession
      << new ClassSessionError
            ;
}

ShmemModule::~ShmemModule() {}

Service* ShmemModule::createService( const String& name )
{
   if (name == "Session" )
   {
      return new SessionService(this);
   }

   return 0;
}

}


FALCON_MODULE_DECL
{
   Falcon::Module* mod = new Falcon::ShmemModule;
   return mod;
}

/* end of shmem.cpp */

