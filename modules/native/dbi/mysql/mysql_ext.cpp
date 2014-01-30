/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_ext.cpp
 *
 * MySQL Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi_common/dbi_driverclass.cpp"

#include "mysql_mod.h"
#include "mysql_ext.h"

/*#
   @beginmodule dbi.mysql
*/
namespace Falcon
{
namespace Ext
{

/*#
   @class MySQL
   @brief Direct interface to MySQL database.
   @param connect String containing connection parameters.

   The connect string uses the standard connection values:
   - uid: user id
   - pwd: password
   - db: database where to connect
   - host: host where to connect (defaults to localhost)
   - port: prot where to connect (defaults to mysql standard port)

   Other than that, mysql presents the following driver-specific parameters
   - socket: UNIX socket name for UNIX-socket based MySQL connections.
*/


ClassMySQLDBIHandle::ClassMySQLDBIHandle():
         ClassDriverDBIHandle("MySQL")
{
}


ClassMySQLDBIHandle::~ClassMySQLDBIHandle()
{
}


void* ClassMySQLDBIHandle::createInstance() const
{
   DBIHandleMySQL* h = new DBIHandleMySQL(this);
   return h;
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

