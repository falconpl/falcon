/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_fm.cpp

   SQLite3 driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "sqlite3_mod.h"
#include "sqlite3_ext.h"
#include "version.h"

#include <falcon/module.h>

// Instantiate the driver service
Falcon::DBIServiceSQLite3 theSQLite3Service;

/*#
   @module sqlite3 Sqlite driver module
   @brief DBI extension supporting sqlite3 embedded database engine
*/

FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "sqlite3" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the DBI module.
   self->addDepend( "dbi", "dbi", true, false );

   // also, we declare a SQLite3 class, which derives from DBIHandler which
   // is in the DBI module.
   Falcon::Symbol *dbh_class = self->addExternalRef( "dbi.%Handle" ); // it's external
   dbh_class->imported( true );
   Falcon::Symbol *sqlite3_class = self->addClass( "SQLite3", Falcon::Ext::SQLite3_init )
      ->addParam("connect")->addParam("options");
   sqlite3_class->getClassDef()->addInheritance( new Falcon::InheritDef( dbh_class ) );
   sqlite3_class->setWKS( true );

   // service publication
   self->publishService( &theSQLite3Service );

   return self;
}

/* end of sqlite3_fm.cpp */
