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

FALCON_FUNC MySQL_init( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   Item *i_tropts = vm->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_tropts != 0 && ! i_tropts->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S,[S]" ) );
   }

   String *params = paramsI->asString();

   DBIHandle *hand = 0;
   try
   {
      hand = theMySQLService.connect( *params );
      if( i_tropts != 0 )
      {
         hand->options( *i_tropts->asString() );
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = theMySQLService.makeInstance( vm, hand );
      vm->retval( instance );
   }
   catch( DBIError* error )
   {
      delete hand;
      throw error;
   }
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

