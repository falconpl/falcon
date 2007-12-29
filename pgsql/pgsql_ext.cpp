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
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include <falcon/engine.h>
#include <libpq-fe.h>

#include "pgsql.h"
#include "pgsql_ext.h"

namespace Falcon
{
namespace Ext
{

FALCON_FUNC PgSQL_init( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   DBIHandlePgSQL *dbh = static_cast<DBIHandlePgSQL *>(
      thePgSQLService.connect( "", false, status, connectErrorMessage ) );
   
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

/* end of pgsql_ext.cpp */

