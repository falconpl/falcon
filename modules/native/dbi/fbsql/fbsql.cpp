/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql.cpp
 *
 * Firebird driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * Firebird module.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 20 Sep 2010 21:02:16 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "fbsql_mod.h"
#include "fbsql_ext.h"
#include "version.h"

// Instantiate the driver service
Falcon::DBIServiceFirebird theFirebirdService;

/*#
   @module mysql MySQL driver module
   @brief DBI extension supporting MySQL
*/

// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "fbsql" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( "dbi", "dbi", true, false );

   Falcon::Symbol *dbh_class = self->addExternalRef( "dbi.%Handle" ); // it's external
   dbh_class->imported( true );
   Falcon::Symbol *firebird_class = self->addClass( "FirebirdSQL", Falcon::Ext::Firebird_init );
   firebird_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   firebird_class->setWKS( true );

   // service publication
   self->publishService( &theFirebirdService );

   // we're done
   return self;
}

/* end of fbsql.cpp */

