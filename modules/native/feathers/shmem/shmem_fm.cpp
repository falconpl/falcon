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

#include "shmem_ext.h"
#include "session_ext.h"
#include "session_srv.h"
#include "ipsem_ext.h"
#include "errors.h"

#include "version.h"

#include "shmem_fm.h"

namespace Falcon {
namespace Feathers {


// initialize the module
ModuleShmem::ModuleShmem():
   Module(FALCON_FEATHER_SHMEM_NAME)
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

ModuleShmem::~ModuleShmem() {}

Service* ModuleShmem::createService( const String& name )
{
   if (name == "Session" )
   {
      return new SessionService(this);
   }

   return 0;
}

}
}

/* end of shmem_fm.cpp */

