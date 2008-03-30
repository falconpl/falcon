/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql.cpp
 *
 * Pgsql driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * MySQL module.
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

#include "mysql_mod.h"
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
   Falcon::Symbol *dbh_class = self->addExternalRef( "%DBIHandle" ); // it's external
   Falcon::Symbol *mysql_class = self->addClass( "MySQL", Falcon::Ext::MySQL_init );
   mysql_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   mysql_class->setWKS( true );

   // we don't have extra functions for the dbhandler of mysql. If whe had,
   // this would be the right place to store them.

   // service publication
   self->publishService( &theMySQLService );

   // we're done
   return self;
}

/* end of mysql.cpp */

