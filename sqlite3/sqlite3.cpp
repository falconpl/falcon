/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3.cpp
 *
 * SQLite3 driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * SQLite3 module.
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "sqlite3_mod.h"
#include "sqlite3_ext.h"
#include "version.h"

// Instantiate the driver service
Falcon::DBIServiceSQLite3 theSQLite3Service;
/*#
   @module sqlite3 Sqlite driver module
   @brief DBI extension supporting sqlite3 embedded database engine
*/
// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "sqlite3" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( "dbi" );

   // also, we declare a SQLite3 class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "%DBIHandle" ); // it's external
   Falcon::Symbol *sqlite3_class = self->addClass( "SQLite3", Falcon::Ext::SQLite3_init );
   sqlite3_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   sqlite3_class->setWKS( true );

   // we don't have extra functions for the dbhandler of mysql. If whe had,
   // this would be the right place to store them.

   // service publication
   self->publishService( &theSQLite3Service );

   // we're done
   return self;
}

/* end of sqlite3.cpp */

