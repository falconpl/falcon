/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_fm.cpp

   SQLite3 driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/dbi/sqlite3/sqlite3_fm.cpp"

#include "sqlite3_mod.h"
#include "sqlite3_ext.h"
#include "version.h"

#include "sqlite3_fm.h"

#include <falcon/dbi_handle.h>

namespace Falcon {

Sqlite3DBIModule::Sqlite3DBIModule():
         DriverDBIModule("dbi.sqlite3")
{
   m_driverDBIHandle = new Ext::ClassSqlite3DBIHandle;
   *this
      << m_driverDBIHandle;
}


Sqlite3DBIModule::~Sqlite3DBIModule()
{
}

}

/*#
   @module dbi.sqlite3 Sqlite driver module
   @brief DBI extension supporting sqlite3 embedded database engine

   Directly importable as @b dbi.sqlite3, it is usually loaded through
   the @a dbi module.
*/

FALCON_MODULE_DECL
{
   return new Falcon::Sqlite3DBIModule;
}

/* end of sqlite3_fm.cpp */
