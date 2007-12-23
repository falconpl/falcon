/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_ext.cpp

   DBI Falcon extension interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 22:02:37 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/engine.h>
#include "dbi.h"
#include "dbi_ext.h"
#include "../include/dbiservice.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC DBIConnect( VMachine *vm )
{
   Item *connParams = vm->param(0);
   // I leave to you the parameter checking and parsing for exercice... :-)
   String provName = "mysql";

   DBIService *provider = theDBIService.loadDbProvider( vm, provName );
   if ( provider != 0 )
   {
      // if it's 0, the service has already raised an error in the vm and we have nothing to do.
      DBIService::dbi_status status;
      DBIHandle *hand = provider->connect( *connParams->asString(), false, status ); // or use the parsed part.
      if ( hand == 0 )
      {
         // raise an error depending on status
         return;
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = provider->makeInstance( vm, hand );
      vm->retval( instance );
   }

   // no matter what we return if we had an error.
}


/**********************************************************
   Handler class
**********************************************************/
FALCON_FUNC DBIHandle_startTransaction( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   DBITransaction *trans = dbh->startTransaction();
   if ( trans == 0 )
   {
      // raise an error depending on dbh->getLastError();
      return;
   }

   Item *trclass = vm->findGlobalItem( "%DBITransaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}

FALCON_FUNC DBIHandle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
   // todo: raise on error
}

/**********************************************************
   Transaction class
**********************************************************/

FALCON_FUNC DBITransaction_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbh = static_cast<DBITransaction *>( self->getUserData() );

   Item *i_query = vm->param(0);
   if( i_query == 0 || ! i_query->isString() )
   {
      // raise error
      return;
   }

   if ( dbh->query( *i_query->asString() ) != DBITransaction::s_ok )
   {
      // raise error
   }

   vm->retval(0); // or anything you want to return
}



}
}

/* end of dbi_ext.cpp */

