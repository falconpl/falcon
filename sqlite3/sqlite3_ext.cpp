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

namespace Falcon
{
namespace Ext
{

FALCON_FUNC SQLite3_init( VMachine *vm )
{
      Item *i_connParams = vm->param(0);
   if ( i_connParams != 0 && ! i_connParams->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                          .extra("[S]") ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   dbi_status status;
   String connectErrorMessage;
   const String& params = i_connParams == 0 ? String("") : *i_connParams->asString();
   
   DBIHandleSQLite3 *dbh = static_cast<DBIHandleSQLite3 *>(
      theSQLite3Service.connect( params, false, status, connectErrorMessage ) );
   
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

/* end of sqlite3_ext.cpp */

