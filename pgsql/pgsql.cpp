/*
   FALCON - The Falcon Programming Language.
   FILE: pgsql.cpp
   
   Pgsql driver main module
   This is BOTH a driver for the DBI interface AND a standalone
   PGSQL module.
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: Sun Dec 23 21:45:01 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include "pgsql.h"
#include "version.h"
#include "pgsql_ext.h"

// Instantiate the driver service
Falcon::DBIServicePgSQL thePgSQLService;

// the main module
FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "pgsql" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( self->addString("dbi") );

   // also, we declare a MySQL class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "%DBIHandler" ); // it's external
   Falcon::Symbol *pgsql_class = self->addClass( "PgSQL", Falcon::Ext::PgSQL_init );
   pgsql_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );

   // we don't have extra functions for the dbhandler of mysql. If whe had,
   // this would be the right place to store them.

   // service pubblication
   self->publishService( &thePgSQLService );

   // we're done
   return self;
}

/* end of pgsql.cpp */

