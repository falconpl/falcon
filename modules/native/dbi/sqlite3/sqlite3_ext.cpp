/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.cpp
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>
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

FALCON_FUNC SQLite3_init( VMachine *vm )
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

/* end of sqlite3_ext.cpp */
