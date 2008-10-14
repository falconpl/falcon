/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_ext.cpp
 *
 * MySQL Falcon extension interface
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

namespace Falcon
{
namespace Ext
{

FALCON_FUNC ODBC_init( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   DBIHandleODBC *dbh = static_cast<DBIHandleODBC *>(
      theMySQLService.connect( "", false, status, connectErrorMessage ) );
   
   if ( dbh == 0 )
   {
      if ( connectErrorMessage.length() == 0 ) 
         connectErrorMessage = "An unknown error has occurred during connect";
      
      vm->raiseModError( new DBIError( ErrorParam( status, __LINE__ )
                                       .desc( connectErrorMessage ) ) );
      return ;
   }
   
   self->setUserData( dbh );
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of mysql_ext.cpp */

