/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_fm.cpp
 *
 * PgSQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar, Stanislas Marquis
 * Begin: Sun Dec 23 21:54:42 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "pgsql_ext.h"
#include "pgsql_mod.h"
#include "version.h"

/*#
   @module pgsql Postgre SQL database driver module
   @brief DBI extension supporting Postgre SQL database
*/

// Instantiate the driver service
Falcon::DBIServicePgSQL thePgSQLService;

// the main module
FALCON_MODULE_DECL
{
    // Module declaration
    Falcon::Module *self = new Falcon::Module();
    self->name( "pgsql" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    // first of all, we need to declare our dependency from the DBI module.
    self->addDepend( "dbi", "dbi", true, false );

    // also, we declare a PgSQL class, which derives from DBIHandler which
    // is in the DBI module.
    Falcon::Symbol *dbh_class = self->addExternalRef( "dbi.%Handle" ); // it's external
    dbh_class->imported( true );
    Falcon::Symbol *pgsql_class = self->addClass( "PgSQL", Falcon::Ext::PgSQL_init );
    pgsql_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
    pgsql_class->setWKS( true );

    // we don't have extra functions for the dbhandler of mysql. If whe had,
    // this would be the right place to store them.

    // named prepared statements
    self->addClassMethod( pgsql_class, "prepareNamed", Falcon::Ext::PgSQL_prepareNamed )
            .asSymbol()->addParam( "name" )->addParam( "query" );

    // service publication
    self->publishService( &thePgSQLService );

    // we're done
    return self;
}
