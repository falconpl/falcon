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

   
   Other than the base DBI class options, this class supports
   the following options:

   - bigint (on/off): By default, the ODBC drivers can't deal
     with int64 (64 bit integers) data. Setting this on, it is
     possible to send int64 data through prepared statements.
*/


/*#
   @class ODBC
   @brief Connects to a ODBC database.
   @param params ODBC Connection parameters.
   @optparam options Default statement options for this connection.

   The @b connect string is directly passed to the ODBC driver
   for connection, so it must respect ODBC standards and specific
   extensions of the target database.
*/

FALCON_FUNC ODBC_init( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   Item *i_connParams = vm->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_connParams != 0 && ! i_connParams->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S,[S]" ) );
   }

   CoreObject *self = vm->self().asObject();
   String connectErrorMessage;

   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandle *hand = 0;

   try
   {
      hand = theODBCService.connect( params );
      if( i_connParams != 0 )
      {
         hand->options( *i_connParams->asString() );
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = theODBCService.makeInstance( vm, hand );
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

/* end of odbc_ext.cpp */

