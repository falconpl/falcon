/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_srv.cpp
 *
 * SQLite3 Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>

#include <falcon/engine.h>
#include "sqlite3_mod.h"

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetSQLite3::DBIRecordsetSQLite3( DBIHandle *dbh, sqlite3_stmt *res )
    : DBIRecordset( dbh )
{
   m_res = res;

   m_row = -1; // BOF
   m_columnCount = sqlite3_data_count( res );
}

DBIRecordsetSQLite3::~DBIRecordsetSQLite3()
{
   if ( m_res != NULL )
      close();
}

dbi_type DBIRecordsetSQLite3::getFalconType( int typ )
{
   switch ( typ )
   {
   case SQLITE_INTEGER: // TODO: no 32/64bit, which should I return?
      return dbit_integer64;

   case SQLITE_FLOAT:
      return dbit_numeric;

   default:
      return dbit_string;
   }
}

dbi_status DBIRecordsetSQLite3::next()
{
   int res = sqlite3_step( m_res );
   switch ( res )
   {
   case SQLITE_DONE:
      return dbi_eof;

   case SQLITE_ROW:
      m_row++;
      if ( m_columnCount == 0 )
         m_columnCount = sqlite3_data_count( m_res );
      return dbi_ok;

   default:
      return dbi_error;
   }
}

int DBIRecordsetSQLite3::getColumnCount()
{
   return m_columnCount;
}

dbi_status DBIRecordsetSQLite3::getColumnNames( char *names[] )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      names[cIdx] = (char *) sqlite3_column_name( m_res, cIdx );

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::getColumnTypes( dbi_type *types )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      types[cIdx] = getFalconType( sqlite3_column_type( m_res, cIdx ) );

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asString( const int columnIndex, String &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   const char *v = (const char *) sqlite3_value_text( sv );

   value = String( v );
   value.bufferize();

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asBoolean( const int columnIndex, bool &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   const char *v = (const char *) sqlite3_value_text( sv );

   if (strncmp( v, "t", 1 ) == 0 || strncmp( v, "T", 1 ) == 0 || strncmp( v, "1", 1 ) == 0)
      value = true;
   else
      value = false;

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asInteger( const int columnIndex, int32 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   value = sqlite3_value_int( sv );

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asInteger64( const int columnIndex, int64 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   value = sqlite3_value_int64( sv );

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asNumeric( const int columnIndex, numeric &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   value = sqlite3_value_double( sv );

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asDate( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   const char *v = (const char *) sqlite3_value_text( sv );
   String tv( v );

   // 2007-12-27
   // 0123456789

   int64 year, month, day;
   tv.subString( 0, 4 ).parseInt( year );
   tv.subString( 5, 7 ).parseInt( month );
   tv.subString( 8, 10 ).parseInt( day );

   value.m_year = year;
   value.m_month = month;
   value.m_day = day;
   value.m_hour = 0;
   value.m_minute = 0;
   value.m_second = 0;
   value.m_msec = 0;

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   const char *v = (const char *) sqlite3_value_text( sv );
   String tv( v );

   // 01:02:03
   // 01234567

   int64 hour, minute, second;
   tv.subString( 0, 2 ).parseInt( hour );
   tv.subString( 3, 5 ).parseInt( minute );
   tv.subString( 6, 8 ).parseInt( second );

   value.m_year = 0;
   value.m_month = 0;
   value.m_day = 0;
   value.m_hour = hour;
   value.m_minute = minute;
   value.m_second = second;
   value.m_msec = 0;

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asDateTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;

   sqlite3_value *sv = sqlite3_column_value( m_res, columnIndex );
   if ( sqlite3_value_type( sv ) == SQLITE_NULL )
      return dbi_nil_value;

   const char *v = (const char *) sqlite3_value_text( sv );
   String tv( v );

   // 2007-10-20 01:02:03
   // 0123456789012345678

   int64 year, month, day, hour, minute, second;
   tv.subString(  0,  4 ).parseInt( year );
   tv.subString(  5,  7 ).parseInt( month );
   tv.subString(  8, 10 ).parseInt( day );
   tv.subString( 11, 13 ).parseInt( hour );
   tv.subString( 14, 16 ).parseInt( minute );
   tv.subString( 17, 19 ).parseInt( second );

   value.m_year = year;
   value.m_month = month;
   value.m_day = day;
   value.m_hour = hour;
   value.m_minute = minute;
   value.m_second = second;
   value.m_msec = 0;

   return dbi_ok;
}

int DBIRecordsetSQLite3::getRowCount()
{
   return -1; // SQLite3 will not tell us how many rows in a result set
}

int DBIRecordsetSQLite3::getRowIndex()
{
   return m_row;
}

void DBIRecordsetSQLite3::close()
{
   if ( m_res != NULL ) {
      sqlite3_finalize( m_res );
      m_res = NULL;
   }
}

dbi_status DBIRecordsetSQLite3::getLastError( String &description )
{
   sqlite3 *conn = ( (DBIHandleSQLite3 *) m_dbh )->getConn();

   if ( conn == NULL )
      return dbi_invalid_connection;

   const char *errorMessage = sqlite3_errmsg( conn );

   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionSQLite3::DBITransactionSQLite3( DBIHandle *dbh )
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionSQLite3::query( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   sqlite3 *conn = ((DBIHandleSQLite3 *) m_dbh)->getConn();
   sqlite3_stmt *res;
   const char *unusedSQL;
   int status = sqlite3_prepare( conn, asQuery.c_str(), asQuery.length(), &res, &unusedSQL );

   if ( res == NULL ) {
      retval = dbi_memory_allocation_error;
      return NULL;
   }

   switch ( status )
   {
   case SQLITE_OK:
      retval = dbi_ok;
      break;

   default: // TODO: return more useful information than this!
      retval = dbi_error;
      break;
   }

   if ( retval != dbi_ok )
      return NULL;

   return new DBIRecordsetSQLite3( m_dbh, res );
}

int DBITransactionSQLite3::execute( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   sqlite3 *conn = ((DBIHandleSQLite3 *) m_dbh)->getConn();
   sqlite3_stmt *res;
   const char *unusedSQL;
   int status = sqlite3_prepare( conn, asQuery.c_str(), asQuery.length(), &res, &unusedSQL );

   if ( res == NULL ) {
      retval = dbi_memory_allocation_error;
      return 0;
   }

   if ( status == SQLITE_OK ) {
      status = sqlite3_step( res ); // execute the actual statement
   }

   int affectedRows;

   switch ( status )
   {
   case SQLITE_OK:
   case SQLITE_DONE:
      affectedRows = sqlite3_changes( conn );
      retval = dbi_ok;
      break;

   default: // TODO: provide better error info than this!
      retval = dbi_execute_error;
      affectedRows = -1;
      break;
   }

   sqlite3_finalize( res );

   return affectedRows;
}

dbi_status DBITransactionSQLite3::begin()
{
   dbi_status retval;

   execute( "BEGIN", retval );

   if ( retval == dbi_ok )
      m_inTransaction = true;

   return retval;
}

dbi_status DBITransactionSQLite3::commit()
{
   dbi_status retval;

   execute( "COMMIT", retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionSQLite3::rollback()
{
   dbi_status retval;

   execute( "ROLLBACK", retval );

   m_inTransaction = false;

   return retval;
}

void DBITransactionSQLite3::close()
{
   // TODO: return a status code here because of the potential commit
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;

   m_dbh->closeTransaction( this );
}

dbi_status DBITransactionSQLite3::getLastError( String &description )
{
   return dbi_ok;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBITransaction *DBIHandleSQLite3::startTransaction()
{
   DBITransactionSQLite3 *t = new DBITransactionSQLite3( this );
   if ( t->begin() != dbi_ok ) {
      // TODO: filter useful information to the script level
      delete t;

      return NULL;
   }

   return t;
}

DBIHandleSQLite3::DBIHandleSQLite3()
{
   m_conn = NULL;
   m_connTr = NULL;
}

DBIHandleSQLite3::DBIHandleSQLite3( sqlite3 *conn )
{
   m_conn = conn;
   m_connTr = NULL;
}

dbi_status DBIHandleSQLite3::closeTransaction( DBITransaction *tr )
{
   return dbi_ok;
}

DBIRecordset *DBIHandleSQLite3::query( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionSQLite3( this );
   }

   return m_connTr->query( sql, retval );
}

int DBIHandleSQLite3::execute( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionSQLite3( this );
   }

   return m_connTr->execute( sql, retval );
}

int64 DBIHandleSQLite3::getLastInsertedId()
{
   // PostgreSQL requires a sequence name
   return sqlite3_last_insert_rowid( m_conn );
}

int64 DBIHandleSQLite3::getLastInsertedId( const String& sequenceName )
{
   // SQLite3 does not support insert id's by name
   return sqlite3_last_insert_rowid( m_conn );
}

dbi_status DBIHandleSQLite3::getLastError( String &description )
{
   if ( m_conn == NULL )
      return dbi_invalid_connection;

   const char *errorMessage = sqlite3_errmsg( m_conn );
   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

dbi_status DBIHandleSQLite3::escapeString( const String &value, String &escaped )
{
   if ( value.length() == 0 )
      return dbi_ok;

   AutoCString asValue( value );

   char *cTo = sqlite3_mprintf( "%q", asValue.c_str() );
   escaped = cTo;
   escaped.bufferize();

   sqlite3_free( cTo );

   return dbi_ok;
}

dbi_status DBIHandleSQLite3::close()
{
   if ( m_conn != NULL ) {
      sqlite3_close( m_conn );
      m_conn = NULL;
   }

   return dbi_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

dbi_status DBIServiceSQLite3::init()
{
   return dbi_ok;
}

DBIHandle *DBIServiceSQLite3::connect( const String &parameters, bool persistent,
                                     dbi_status &retval, String &errorMessage )
{
   AutoCString connParams( parameters );
   sqlite3 *conn;
   int result = sqlite3_open( connParams.c_str(), &conn );
   if ( conn == NULL ) {
      retval = dbi_memory_allocation_error;
      return NULL;
   } else if ( result != SQLITE_OK ) {
      retval = dbi_connect_error;
      errorMessage = sqlite3_errmsg( conn );
      errorMessage.bufferize();

      sqlite3_close( conn );

      return NULL;
   }

   retval = dbi_ok;

   return new DBIHandleSQLite3( conn );
}

CoreObject *DBIServiceSQLite3::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "SQLite3" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "SQLite3" ) {
      vm->raiseModError( new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                                      .desc( "SQLite3 DBI driver was not found" ) ) );
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of sqlite3_srv.cpp */

