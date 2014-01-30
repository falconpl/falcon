/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.cpp
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Thu, 30 Jan 2014 13:47:51 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi_common/dbi_driverclass.cpp"

#include <sqlite3.h>
#include "sqlite3_mod.h"
#include "sqlite3_ext.h"

/*#
   @beginmodule dbi.sqlite3
*/
namespace Falcon
{
namespace Ext
{

/*#
   @class SQLite3
   @brief Direct interface to SQLite3 database.
   @param connect String containing connection parameters.
   @optparam options Default statement options for this connection.
*/


ClassSqlite3DBIHandle::ClassSqlite3DBIHandle():
         ClassDriverDBIHandle("Sqlite3")
{
}


ClassSqlite3DBIHandle::~ClassSqlite3DBIHandle()
{
}


void* ClassSqlite3DBIHandle::createInstance() const
{
   DBIHandleSQLite3* h = new DBIHandleSQLite3(this);
   return h;
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of sqlite3_ext.cpp */
