/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql.cpp
 *
 * Mysql driver main module
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
 */

#include "mysql_mod.h"
#include "version.h"
#include "mysql_ext.h"

// Instantiate the driver service
Falcon::DBIServiceMySQL theMySQLService;

/*#
   @module dbi.mysql MySQL driver module
   @brief DBI extension supporting MySQL


   Directly importable as @b dbi.mysql, it is usually loaded through
   the @a dbi module.
*/

// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "mysql" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( "dbi", "dbi", true, false );

   // also, we declare a MySQL class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "dbi.%Handle" ); // it's external
   dbh_class->imported( true );
   Falcon::Symbol *mysql_class = self->addClass( "MySQL", Falcon::Ext::MySQL_init );
   mysql_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   mysql_class->setWKS( true );

   // we don't have extra functions for the dbhandler of mysql. If we had,
   // this would be the right place to store them.

   // service publication
   self->publishService( &theMySQLService );

   // we're done
   return self;
}

/* end of mysql.cpp */

