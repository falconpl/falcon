/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql.cpp
 *
 * Mysql driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * MySQL module.
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "mysql_mod.h"
#include "version.h"
#include "mysql_ext.h"
#include "mysql_fm.h"

/*#
   @module dbi.mysql MySQL driver module
   @brief DBI extension supporting MySQL

   Directly importable as @b dbi.mysql, it is usually loaded through
   the @a dbi module.
*/

namespace Falcon
{

ModuleMySQL::ModuleMySQL():
         DriverDBIModule("dbi.mysql")
{
   m_driverDBIHandle = new Ext::ClassMySQLDBIHandle;

   *this
      << m_driverDBIHandle;
}

ModuleMySQL::~ModuleMySQL()
{
}

}

// the main module
FALCON_MODULE_DECL
{
   // we're done
   return new Falcon::ModuleMySQL;
}

/* end of mysql.cpp */

