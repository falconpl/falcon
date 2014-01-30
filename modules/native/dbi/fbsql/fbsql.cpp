/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql.cpp
 *
 * Firebird driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * Firebird module.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 20 Sep 2010 21:02:16 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "fbsql_mod.h"
#include "fbsql_ext.h"
#include "fbsql_fm.h"
#include "version.h"

/*#
   @module dbi.fbsql Firebird Database driver module
   @brief DBI extension supporting Firebird


   Directly importable as @b dbi.fbsql, it is usually loaded through
   the @a dbi module.
*/

namespace Falcon
{

ModuleFBSQL::ModuleFBSQL():
   DriverDBIModule("dbi.fbsql")
{
   m_driverDBIHandle = new Ext::ClassFBSQLDBIHandle;

   *this
      << m_driverDBIHandle;
}


ModuleFBSQL::~ModuleFBSQL()
{
}

}

// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::ModuleFBSQL;
   // we're done
   return self;
}

/* end of fbsql.cpp */

