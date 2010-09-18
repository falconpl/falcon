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
#include <falcon/sys.h>
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

   int flags = SQLITE_OPEN_READWRITE;
   if( connParams.m_sCreate == "always" )
   {
      flags |= SQLITE_OPEN_CREATE;

      // sqlite3 doesn't drop databases: delete files.
      int32 fsStatus;
      FileStat::e_fileType st;
      if ( Sys::fal_fileType( connParams.m_szDb, st ) &&
           ! Sys::fal_unlink( connParams.m_szDb, fsStatus ) )
      {
         throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT_CREATE, __LINE__)
                      .extra( parameters )
                   );
      }
   }
   else if ( connParams.m_sCreate == "cond" )
   {
      flags |= SQLITE_OPEN_CREATE;
   }
   else if( connParams.m_sCreate != "" )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
              .extra( parameters )
           );
   }

   sqlite3 *conn;
   int result = sqlite3_open_v2( connParams.m_szDb, &conn, flags, NULL );

   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
   }
   else if ( result == SQLITE_CANTOPEN )
   {
      int er = connParams.m_sCreate == "cond" ?
               FALCON_DBI_ERROR_CONNECT_CREATE : FALCON_DBI_ERROR_DB_NOTFOUND;

      throw new DBIError( ErrorParam( er, __LINE__)
                    .extra( sqlite3_errmsg( conn ) )
                 );
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

