/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_ext.cpp
 *
 * ODBC Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin:
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>

#include "odbc_mod.h"
#include "odbc_ext.h"

/*#
   @beginmodule dbi.odbc
*/
namespace Falcon
{
namespace Ext
{

/*#
   @class ODBC
   @brief Interface to ODBC connections.
   @param connect String containing connection parameters.
   @optparam options Connection and query default options.

   The ODBC drivers have a limited ability to determine
   the underlying database types; for this reason, it's advisable
   to limit the usage of prepared statements, and rely on @b query,
   which performs safer verbatim parameter expansion.

   The @b connect string is directly passed to the ODBC driver
   for connection, so it must respect ODBC standards and specific
   extensions of the target database.

   Other than the base DBI class options, this class supports
   the following options:

   - bigint (on/off): By default, the ODBC drivers can't deal
     with int64 (64 bit integers) data. Setting this on, it is
     possible to send int64 data through prepared statements.
*/

   
ClassODBCDBIHandle::ClassODBCDBIHandle():
         ClassDriverDBIHandle("ODBC")
{
}


ClassODBCDBIHandle::~ClassODBCDBIHandle()
{
}


void* ClassODBCDBIHandle::createInstance() const
{
   DBIHandleODBC* h = new DBIHandleODBC(this);
   return h;
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of odbc_ext.cpp */

