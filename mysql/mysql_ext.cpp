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

#include <falcon/engine.h>

#include "mysql_mod.h"
#include "mysql_ext.h"

/*#
   @beginmodule mysql
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

FALCON_FUNC MySQL_init( VMachine *vm )
{
   Item *i_connParams = vm->param(0);
   if ( i_connParams != 0 && ! i_connParams->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                          .extra("[S]") );
      return;
   }

   CoreObject *self = vm->self().asObject();
   //dbi_status status;
   String connectErrorMessage;
   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandleMySQL *dbh = static_cast<DBIHandleMySQL *>(
      theMySQLService.connect( params, false ) );
   // it's the service that must throw on error.
   fassert( dbh != 0 );
   self->setUserData( dbh );
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

