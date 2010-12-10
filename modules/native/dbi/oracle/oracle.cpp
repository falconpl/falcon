/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle.cpp
 *
 * Oracle driver main module
 *
 * This is BOTH a driver for the DBI interface AND a standalone
 * Oracle module.
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "oracle_mod.h"
#include "version.h"
#include "oracle_ext.h"

// Instantiate the driver service
Falcon::DBIServiceOracle theOracleService;

/*#
   @module oracle Oracle driver module
   @brief DBI extension supporting Oracle
*/

// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "oracle" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( "dbi", "dbi", true, false );

   // also, we declare an Oracle class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "dbi.%Handle" ); // it's external
   Falcon::Symbol *oracle_class = self->addClass( "Oracle", Falcon::Ext::Oracle_init );
   oracle_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   oracle_class->setWKS( true );

   // service publication
   self->publishService( &theOracleService );

   // we're done
   return self;
}

/* end of oracle.cpp */

