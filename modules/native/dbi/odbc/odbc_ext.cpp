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
   @beginmodule odbc
*/
namespace Falcon
{
namespace Ext
{

/*#
      @class ODBC
      @brief Direct interface to ODBC connections.
      @param connect String containing connection parameters.
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
   Item *i_tropts = vm->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_tropts != 0 && ! i_tropts->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S,[S]" ) );
   }

   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;

   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandle *hand = 0;

   try
   {
      hand = theSQLite3Service.connect( *params );
      if( i_tropts != 0 )
      {
         hand->options( *i_tropts->asString() );
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = theSQLite3Service.makeInstance( vm, hand );
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

