/* 
   FALCON - The Falcon Programming Language.
   FILE: odbc_fm.cpp
 
   ODBC driver for DBI - main module
 
   This is BOTH a driver for the DBI interface AND a standalone
   ODBC module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue Sep 30 17:00:00 2008
  
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
  
   See LICENSE file for licensing details.
*/

#include "odbc_mod.h"
#include "version.h"
#include "odbc_ext.h"
#include "odbc_fm.h"

/*#
   @module dbi.odbc ODBC driver module.
   @brief DBI extension supporting ODBC connections.


   Directly importable as @b dbi.odbc, it is usually loaded through
   the @a dbi module.
*/
namespace Falcon {

ODBCDBIModule::ODBCDBIModule():
         DriverDBIModule("dbi.odbc")
{
   m_driverDBIHandle = new Ext::ClassODBCDBIHandle;
   *this
      << m_driverDBIHandle;
}


ODBCDBIModule::~ODBCDBIModule()
{
}

}

// the main module
FALCON_MODULE_DECL
{
   Falcon::Module* self = new Falcon::ODBCDBIModule;

   // we're done
   return self;
}

/* end of odbc_fm.cpp */
