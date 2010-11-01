/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_ext.cpp
 *
 * PgSQL Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Sun Dec 23 21:51:18 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>
#include <libpq-fe.h>

#include "pgsql_mod.h"
#include "pgsql_ext.h"

/*#
   @beginmodule pgsql
*/
namespace Falcon
{
namespace Ext
{

/*#
      @class PgSQL
      @brief Direct interface to Postgre SQL database.
      @param connect String containing connection parameters.
*/


/*#
   @init PgSQL
   @brief Connects to a PgSQL database.

   The @b connect string is directly passed to the low level postgre driver.
*/

FALCON_FUNC PgSQL_init( VMachine *vm )
{
#if 0
   Item *i_connParams = vm->param(0);
   if ( i_connParams != 0 && ! i_connParams->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                          .extra("[S]") );
      return;
   }

   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandlePgSQL *dbh = static_cast<DBIHandlePgSQL *>(
      thePgSQLService.connect( params, false, status, connectErrorMessage ) );

   if ( dbh == 0 )
   {
      if ( connectErrorMessage.length() == 0 )
         connectErrorMessage = "An unknown error has occured during connect";

      throw new DBIError( ErrorParam( status, __LINE__ )
                                       .desc( connectErrorMessage ) );
      return ;
   }

   self->setUserData( dbh );
#endif

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
      hand = thePgSQLService.connect( *params );
      /*if( i_tropts != 0 )
      {
         hand->options( *i_tropts->asString() );
      }*/

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = thePgSQLService.makeInstance( vm, hand );
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

/* end of pgsql_ext.cpp */

