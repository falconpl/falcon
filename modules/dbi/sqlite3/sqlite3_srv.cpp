/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_srv.cpp
 *
 * SQLite3 Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 18:23:20 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>

#include <falcon/engine.h>
#include "sqlite3_mod.h"
#include "sqlite3.h"

namespace Falcon
{

/******************************************************************************
 * Main service class
 *****************************************************************************/

DBIServiceSQLite3::DBIServiceSQLite3():
   DBIService( "DBI_sqlite3" )
{}


void DBIServiceSQLite3::init()
{
}

DBIHandle *DBIServiceSQLite3::connect( const String &parameters )
{
   // Parse the connection string.
   DBIConnParams connParams;

   if( ! connParams.parse( parameters ) || connParams.m_szDb == 0 )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .extra( parameters )
      );
   }

   sqlite3 *conn;
   int result = sqlite3_open( connParams.m_szDb, &conn );
   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
   }
   else if ( result != SQLITE_OK )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
              .extra( sqlite3_errmsg( conn ) )
           );
   }

   return new DBIHandleSQLite3( conn );
}

CoreObject *DBIServiceSQLite3::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "SQLite3" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "SQLite3" ) {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__ ) );
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of sqlite3_srv.cpp */

