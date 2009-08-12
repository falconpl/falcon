/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_ext.cpp
 *
 * ODBC Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Tiziano De Rubeis
 * Begin: Tue Sep 30 17:00:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
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
   @init ODBC
   @brief Connects to a ODBC database.


   The @b connect string is directly passed to the ODBC driver
   for connection, so it must respect ODBC standards and specific
   extensions of the target database.
*/

FALCON_FUNC ODBC_init( VMachine *vm )
{
   Item *i_connParams = vm->param(0);
   if ( i_connParams != 0 && ! i_connParams->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                          .extra("[S]") );
   }

   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;

   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();

   DBIHandleODBC *dbh = static_cast<DBIHandleODBC *>(
      theODBCService.connect( params, false, status, connectErrorMessage ) );
   
   if ( dbh == 0 )
   {
      if ( connectErrorMessage.length() == 0 ) 
         connectErrorMessage = "An unknown error has occurred during connect";
      
      throw new DBIError( ErrorParam( status, __LINE__ )
                                       .desc( connectErrorMessage ) );
   }
   
   self->setUserData( dbh );
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

