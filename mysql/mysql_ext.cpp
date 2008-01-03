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
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include <falcon/engine.h>
#include <mysql/mysql.h>

#include "mysql.h"
#include "mysql_ext.h"

namespace Falcon
{
namespace Ext
{

FALCON_FUNC MySQL_init( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   DBIHandleMySQL *dbh = static_cast<DBIHandleMySQL *>(
      theMySQLService.connect( "", false, status, connectErrorMessage ) );
   
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

