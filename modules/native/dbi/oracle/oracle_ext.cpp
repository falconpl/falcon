/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_ext.cpp
 *
 * Oracle Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>

#include "oracle_mod.h"
#include "oracle_ext.h"

/*#
   @beginmodule oracle
*/
namespace Falcon
{
namespace Ext
{

/*#
      @class Oracle
      @brief Direct interface to Oracle database.
      @param connect String containing connection parameters.

*/


/*#
   @init Oracle
   @brief Connects to a Oracle database.

*/

FALCON_FUNC Oracle_init( VMachine *vm )
{
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

   DBIHandleOracle *dbh = static_cast<DBIHandleOracle *>(
      theOracleService.connect( params, false, status, connectErrorMessage ) );

   if ( dbh == 0 )
   {
      if ( connectErrorMessage.length() == 0 )
         connectErrorMessage = "An unknown error has occurred during connect";

      throw new DBIError( ErrorParam( status, __LINE__ )
                                       .desc( connectErrorMessage ) );
      return ;
   }

   self->setUserData( dbh );
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of oracle_ext.cpp */

