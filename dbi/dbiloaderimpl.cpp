/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbiloaderimpl.cpp
 *
 * Implementation of the DBI loader service, used mainly by this module.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 *  Begin: Sun, 23 Dec 2007 20:33:57 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "dbi.h"
#include <falcon/module.h>

namespace Falcon
{

DBILoaderImpl::DBILoaderImpl():
   DBILoader( "DBILOADER" ),
   m_loader("")
{
   m_loader.addFalconPath();
}

DBILoaderImpl::~DBILoaderImpl()
{}

DBIService *DBILoaderImpl::loadDbProvider( VMachine *vm, const String &provName )
{
   DBIService *serv = static_cast<DBIService *>( vm->getService( "DBI_" + provName ) );
   Module *mod = 0;
   if ( serv == 0 )
   {
      // ok, let's try to load the service
      mod = m_loader.loadName( provName );
      try {
         vm->link( mod );
      }
      catch( Error * )
      {
         mod->decref();
         throw;
      }

      // the VM has linked the module, we get rid of it.
      mod->decref();
      //NOTE: we must actually have a local map, as we may be fed with different VMs.
      //We should load only one module for each type, and give to each VM a pre-loaded
      // module, if available.

      // everything went fine; the service is in the vm, but we can access it also
      // from the module
      serv = static_cast<DBIService *>( mod->getService(  "DBI_" + provName ) );

      if ( serv->init() != dbi_ok )
      {
         // we should raise an error here...
         return 0;
      }
   }

   return serv;
}

}

/* end of dbiloaderimpl.cpp */

