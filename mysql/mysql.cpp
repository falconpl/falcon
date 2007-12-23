/*
   FALCON - The Falcon Programming Language.
   FILE: mysql.cpp

   Mysql driver main module
   This is BOTH a driver for the DBI interface AND a
   standalone MYSQL module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 22:50:42 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include "mysql.h"
#include "version.h"
#include "mysql_ext.h"

// Instantiate the driver service
Falcon::DBIServiceMySQL theMySQLService;

// the main module
FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "mysql" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( self->addString("dbi") );

   // also, we declare a MySQL class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "%DBIHandler" ); // it's external
   Falcon::Symbol *mysql_class = self->addClass( "MySQL", Falcon::Ext::MySQL_init );
   mysql_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );

   // we don't have extra functions for the dbhandler of mysql. If whe had,
   // this would be the right place to store them.

   // service pubblication
   self->publishService( &theMySQLService );

   // we're done
   return self;
}


/* end of mysql.cpp */

