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
*/


/*#
   @init MySQL
   @brief Connects to a MySQL database.
   @param connect String containing connection parameters.

   The string is in the following format:
   @code
      <host>,<user>,<passwd>,<db>,<port>,<unixSocket>,<clientFlags>
   @endcode

   All the parameters are optional except for port or user. To pass an empty
   parameter, just use two commas one beside anoter. In example, to connect
   to a database "mydb" at localhost as "root" user without password, use:

   @code
      localost,root,,mydb
   @endcode
*/

FALCON_FUNC MySQL_init( VMachine *vm )
{
   Item *i_connParams = vm->param(0);
   if ( i_connParams != 0 && ! i_connParams->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                          .extra("[S]") ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandleMySQL *dbh = static_cast<DBIHandleMySQL *>(
      theMySQLService.connect( params, false, status, connectErrorMessage ) );

   if ( dbh == 0 )
   {
      if ( connectErrorMessage.length() == 0 )
         connectErrorMessage = "An unknown error has occured during connect";

      vm->raiseModError( new DBIError( ErrorParam( status, __LINE__ )
                                       .desc( connectErrorMessage ) ) );
      return ;
   }

   self->setUserData( dbh );
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

