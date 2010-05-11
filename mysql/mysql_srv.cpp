/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_srv.cpp
 *
 * MySQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <errmsg.h>

#include <falcon/engine.h>
#include "mysql_mod.h"
#include "dbi_mod.h"

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetMySQL::DBIRecordsetMySQL( DBITransaction *dbt, MYSQL_RES *res )
    : DBIRecordset( dbt ),
      m_res( res ),
      m_stmt(0)
{
   m_row = -1; // BOF
   m_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
   m_columnCount = mysql_num_fields( res );
   m_fields = mysql_fetch_fields( res );
}

DBIRecordsetMySQL::DBIRecordsetMySQL( DBITransaction *dbt, MYSQL_STMT *stmt )
    : DBIRecordset( dbt ),
       m_res( 0 ),
       m_stmt(stmt)
{
   m_row = -1; // BOF
   m_rowCount = mysql_stmt_num_rows( stmt ); // Only valid when using mysql_store_result instead of use_result
   m_res = mysql_stmt_result_metadata(stmt);
   m_columnCount = mysql_num_fields( m_res );
   m_fields = mysql_fetch_fields( m_res );
}

DBIRecordsetMySQL::~DBIRecordsetMySQL()
{
   if ( m_res != NULL )
      close();
}

/*
dbi_type DBIRecordsetMySQL::getFalconType( int typ )
{
   switch ( typ )
   {
   case MYSQL_TYPE_TINY:
   case MYSQL_TYPE_SHORT:
   case MYSQL_TYPE_LONG:
   case MYSQL_TYPE_INT24:
   case MYSQL_TYPE_BIT:
   case MYSQL_TYPE_YEAR:
      return dbit_integer;

   case MYSQL_TYPE_LONGLONG:
      return dbit_integer64;

   case MYSQL_TYPE_DECIMAL:
   case MYSQL_TYPE_NEWDECIMAL:
   case MYSQL_TYPE_FLOAT:
   case MYSQL_TYPE_DOUBLE:
      return dbit_numeric;

   case MYSQL_TYPE_DATE:
      return dbit_date;

   case MYSQL_TYPE_TIME:
      return dbit_time;

   case MYSQL_TYPE_DATETIME: // TODO: MYSQL_TYPE_TIMESTAMP ?!?
      return dbit_datetime;

   default:
      return dbit_string;
   }
}
*/

int DBIRecordsetMySQL::getColumnCount()
{
   return m_columnCount;
}

bool DBIRecordsetMySQL::getColumnName( int nCol, String& name )
{
   if( nCol >=0  && nCol < m_columnCount )
   {
      name.fromUTF8( m_fields[nCol].name );
      return true;
   }
   return false;
}


bool DBIRecordsetMySQL::getColumnValue( int nCol, Item& value )
{
   if ( m_row == 0 || nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }
#if 0 // TODO
   switch( m_fields[nCol].type )
   {

   }

   value.fromUTF8( m_rowData[columnIndex] );
#endif
   return true;
}


int64 DBIRecordsetMySQL::getRowCount()
{
   return m_rowCount;
}


int64 DBIRecordsetMySQL::getRowIndex()
{
   return m_row;
}

void DBIRecordsetMySQL::close()
{
   if ( m_res != NULL ) {
      mysql_free_result( m_res );
      m_res = NULL;
   }

   if ( m_stmt != 0 )
   {
      mysql_stmt_close( m_stmt );
      m_stmt = 0;
   }
}


/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionMySQL::DBITransactionMySQL( DBIHandle *dbh, bool bAutoCommit )
    : DBITransaction( dbh, bAutoCommit ),
    m_statement(0)
{
   begin();// which one?
}


DBIRecordset *DBITransactionMySQL::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // if we don't have var params, we can use the standard query, after proper escape.
   if ( params.length() == 0 )
   {
      MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();

      AutoCString asQuery( sql );
      if( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
      }

      if ( mysql_field_count( conn ) > 0 )
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_NOMEM );
         affectedRows = (int64) mysql_affected_rows( conn );

         // Get or use the result?
         MYSQL_RES* res = mysql_store_result( conn );
         return new DBIRecordsetMySQL( this, res );
      }
      else
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY_EMPTY );
      }
   }



}


void DBITransactionMySQL::call( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // if we don't have var params, we can use the standard query, after proper escape.
   if ( params.length() == 0 )
   {
      MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();

      AutoCString asQuery( sql );
      if( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
      }
   }
}


void DBITransactionMySQL::prepare( const String &query )
{

}


void DBITransactionMySQL::execute( const ItemArray& params )
{

}


void DBITransactionMySQL::begin()
{
   int64 dummy;
   ItemArray arr;
   call( "BEGIN", dummy, arr );
   MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();
   mysql_autocommit( conn, m_bAutoCommit );
   m_inTransaction = true;
}




void DBITransactionMySQL::commit()
{
   int64 dummy;
   ItemArray arr;
   call( "COMMIT", dummy, arr );
   m_inTransaction = false;
}

void DBITransactionMySQL::rollback()
{
   int64 dummy;
   ItemArray arr;
   call( "ROLLBACK", dummy, arr );
   m_inTransaction = false;
}

void DBITransactionMySQL::close()
{
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;
}

int64 DBITransactionMySQL::getLastInsertedId( const String& name )
{
    // TODO
    return -1;
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/
DBIHandleMySQL::~DBIHandleMySQL()
{
   DBIHandleMySQL::close();
}

DBITransaction *DBIHandleMySQL::startTransaction( bool bAuto, const String& name )
{
   DBITransactionMySQL *t = new DBITransactionMySQL( this, bAuto );

   try
   {
      t->begin();
   }
   catch(...)
   {
      delete t;
      throw;
   }

   return t;
}

DBIHandleMySQL::DBIHandleMySQL()
{
   m_conn = NULL;
}

DBIHandleMySQL::DBIHandleMySQL( MYSQL *conn )
{
   m_conn = conn;
   // we'll be using UTF-8 charset
   mysql_set_character_set( m_conn, "utf8" );
}

#if 0
int64 DBIHandleMySQL::getLastInsertedId( const String& sequenceName )
{
   return mysql_insert_id( m_conn );
}
#endif

void DBIHandleMySQL::close()
{
   if ( m_conn != NULL ) {
      mysql_close( m_conn );
      m_conn = NULL;
   }
}

void DBIHandleMySQL::throwError( const char* file, int line, int code )
{
   const char *errorMessage = mysql_error( m_conn );
   String extra; // dummy

   if ( errorMessage != NULL )
   {
      String description;
      description.N( (int64) mysql_errno( m_conn ) ).A(": ");
      description.A( errorMessage );
      dbh_throwError( file, line, code, extra );
   }
   else
   {
      dbh_throwError( file, line, code, extra );
   }
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceMySQL::init()
{
}

DBIHandle *DBIServiceMySQL::connect( const String &parameters, bool persistent )
{
   // Parse the connection string.
   DBIConnParams connParams;

   // add MySQL specific parameters
   String sSocket, sFlags;
   const char *szSocket;
   connParams.addParameter( "socket", sSocket, &szSocket );
   connParams.addParameter( "flags", sFlags );

   MYSQL *conn = mysql_init( NULL );

   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__)
            .desc( FAL_STR( dbi_msg_nomem ) )
            );
   }

   if( ! connParams.parse( parameters ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .desc( FAL_STR( dbi_msg_connparams ) )
         .extra( parameters )
      );
   }

   long szFlags = 0;
   // TODO parse flags

   if ( mysql_real_connect( conn,
         connParams.m_szHost,
         connParams.m_szUser,
         connParamsm_szPasswd,
         connParams.m_szDb,
         connParams.m_szPort == 0 ? 0 : atoi( connParams.m_szPort ),
         szSocket, 0 ) == NULL
      )
   {
      String errorMessage = mysql_error( conn );
      errorMessage.bufferize();
      mysql_close( conn );

      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
              .desc( FAL_STR( dbi_msg_connect ) )
              .extra( errorMessage )
           );
   }

   return new DBIHandleMySQL( conn );
}

CoreObject *DBIServiceMySQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "MySQL" );
   if ( cl == 0 || ! cl->isClass() )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__ )
               .desc( FAL_STR( dbi_msg_connect ) ) );
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of mysql_srv.cpp */

