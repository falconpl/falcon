/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_srv.cpp
 *
 * ODBC service/driver
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Wed Oct 13 09:44:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define _WIN32_WINNT 0x0500
#if ! defined( _WIN32_WINNT ) || _WIN32_WINNT < 0x0403
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0403
#endif

#include <string.h>
#include <stdio.h>
#include <Windows.h>
#include <falcon/engine.h>
#include "odbc_mod.h"

#include <sqlext.h>
#include <sqltypes.h>

#include <falcon/autocstring.h>

namespace Falcon
{


/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceODBC::init()
{
}

DBIHandle *DBIServiceODBC::connect( const String &parameters )
{
   AutoCString asConnParams( parameters );

   SQLHENV hEnv;
   SQLHDBC hHdbc;

   RETCODE retcode = SQLAllocHandle (SQL_HANDLE_ENV, NULL, &hEnv);

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
         .extra( "Impossible to allocate the ODBC environment" ));
   }

   retcode = SQLSetEnvAttr( hEnv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER) SQL_OV_ODBC3, SQL_IS_INTEGER );

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
         .extra( "Impossible to notify ODBC that this is an ODBC 3.0 app." ));
   }

   // Allocate ODBC connection handle and connect.
   retcode = SQLAllocHandle( SQL_HANDLE_DBC, hEnv, &hHdbc );

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
         .extra( "Impossible to allocate ODBC connection handle and connect." ));
	   return NULL;
   }

   int nSec = 15;
   SQLSetConnectAttr( hHdbc, SQL_LOGIN_TIMEOUT, (SQLPOINTER)(&nSec), 0 );

   SQLCHAR OutConnStr[MAXBUFLEN];
   short OutConnStrLen = MAXBUFLEN;

   retcode = SQLDriverConnect(
	   hHdbc, 
	   NULL, 
	   (SQLCHAR*)asConnParams.c_str(),
      asConnParams.length(),
	   OutConnStr,
	   MAXBUFLEN, 
	   &OutConnStrLen,
	   SQL_DRIVER_NOPROMPT );

   if( ( retcode != SQL_SUCCESS ) && ( retcode != SQL_SUCCESS_WITH_INFO ) )
   {
	   String errorMessage = 
         String("SQLDriverConnect failed. Reason: ") + DBIHandleODBC::GetErrorMessage( SQL_HANDLE_DBC, hHdbc, FALSE );
	   SQLDisconnect( hHdbc );
	   SQLFreeHandle( SQL_HANDLE_DBC, hHdbc );
	   SQLFreeHandle( SQL_HANDLE_ENV, hEnv );

      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
         .extra( errorMessage ));
	   return NULL;
   }

   ODBCConn* conn = new ODBCConn( hEnv, hHdbc );
   return new DBIHandleODBC( conn );
}


CoreObject *DBIServiceODBC::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "ODBC" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "ODBC" ) {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_BASE, __LINE__ )
                                      .desc( "ODBC DBI driver was not found" ) );
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}


} /* namespace Falcon */

/* end of mysql_srv.cpp */

