/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_fm.cpp
 *
 * PgSQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar, Stanislas Marquis
 * Begin: Sun Dec 23 21:54:42 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "pgsql_ext.h"
#include "pgsql_mod.h"
#include "pgsql_fm.h"
#include "version.h"

/*#
   @module dbi.pgsql Postgre SQL database driver module
   @brief DBI extension supporting Postgre SQL database

   Directly importable as @b dbi.pgsql, it is usually loaded through
   the @a dbi module.

*/
namespace Falcon {

PGSQLDBIModule::PGSQLDBIModule():
         DriverDBIModule("dbi.pgsql")
{
   m_driverDBIHandle = new Ext::ClassPGSQLDBIHandle;
   *this
      << m_driverDBIHandle;
}


PGSQLDBIModule::~PGSQLDBIModule()
{
}

}

// the main module
FALCON_MODULE_DECL
{
    Falcon::Module* self = new Falcon::PGSQLDBIModule;

    // we're done
    return self;
}
