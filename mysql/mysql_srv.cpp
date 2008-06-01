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

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetMySQL::DBIRecordsetMySQL( DBIHandle *dbh, MYSQL_RES *res )
    : DBIRecordset( dbh )
{
   m_res = res;

   m_row = -1; // BOF
   m_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
   m_columnCount = mysql_num_fields( res );
   m_fields = mysql_fetch_fields( res );
}

DBIRecordsetMySQL::~DBIRecordsetMySQL()
{
   if ( m_res != NULL )
      close();
}

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

dbi_status DBIRecordsetMySQL::next()
{
   m_rowData = mysql_fetch_row( m_res );
   if ( m_rowData == NULL ) {
      return dbi_eof;
   } else if ( mysql_num_fields( m_res ) == 0 ) {
      unsigned int err = mysql_errno( ((DBIHandleMySQL *) m_dbh)->getConn() );
      switch ( err )
      {
      case CR_SERVER_LOST:
         return dbi_invalid_connection;

      case CR_UNKNOWN_ERROR:
         return dbi_error; // TODO: provide better error information

      default:
         return dbi_error; // TODO: provide better error information
      }
   }

   m_row++;

   // Fetch lengths of each field so we can later deal with binary values that may contain ZERO
   // or NULL values right in the middle of a string.
   m_fieldLengths = mysql_fetch_lengths( m_res );

   return dbi_ok;
}

int DBIRecordsetMySQL::getColumnCount()
{
   return m_columnCount;
}

dbi_status DBIRecordsetMySQL::getColumnNames( char *names[] )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      names[cIdx] = m_fields[cIdx].name;

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::getColumnTypes( dbi_type *types )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      types[cIdx] = getFalconType( m_fields[cIdx].type );

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asString( const int columnIndex, String &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   // TODO: check proper field encoding and transcode.
   value.fromUTF8( m_rowData[columnIndex] );
   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asBlobID( const int columnIndex, String &value )
{
   return dbi_not_implemented;
}

dbi_status DBIRecordsetMySQL::asBoolean( const int columnIndex, bool &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   char *v = m_rowData[columnIndex];

   if (strncmp( v, "t", 1 ) == 0 || strncmp( v, "T", 1 ) == 0 || strncmp( v, "1", 1 ) == 0)
      value = true;
   else
      value = false;

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asInteger( const int columnIndex, int32 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   value = atoi( m_rowData[columnIndex] );

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asInteger64( const int columnIndex, int64 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   value = atoll( m_rowData[columnIndex] );

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asNumeric( const int columnIndex, numeric &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   value = atof( m_rowData[columnIndex] );

   return dbi_ok;
}

dbi_status DBIRecordsetMySQL::asDate( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   String tv( m_rowData[columnIndex] );

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

dbi_status DBIRecordsetMySQL::asTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   String tv( m_rowData[columnIndex] );

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

dbi_status DBIRecordsetMySQL::asDateTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( m_rowData[columnIndex] == NULL )
      return dbi_nil_value;

   String tv( m_rowData[columnIndex] );

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

int DBIRecordsetMySQL::getRowCount()
{
   return m_rowCount;
}

int DBIRecordsetMySQL::getRowIndex()
{
   return m_row;
}

void DBIRecordsetMySQL::close()
{
   if ( m_res != NULL ) {
      mysql_free_result( m_res );
      m_res = NULL;
   }
}

dbi_status DBIRecordsetMySQL::getLastError( String &description )
{
   MYSQL *conn = ( (DBIHandleMySQL *) m_dbh )->getConn();

   if ( conn == NULL )
      return dbi_invalid_connection;

   const char *errorMessage = mysql_error( conn );

   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionMySQL::DBITransactionMySQL( DBIHandle *dbh )
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionMySQL::query( const String &query, dbi_status &retval )
{
   retval = dbi_ok;

   AutoCString asQuery( query );
   MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();

   if ( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
   {
      switch ( mysql_errno( conn ) )
      {
      case CR_COMMANDS_OUT_OF_SYNC:
         retval = dbi_query_error;
         break;

      case CR_SERVER_GONE_ERROR:
      case CR_SERVER_LOST:
         retval = dbi_invalid_connection;
         break;

      default:
         retval = dbi_error;
      }
      return NULL;
   }

   if ( mysql_field_count( conn ) > 0 )
   {
      MYSQL_RES* res = mysql_store_result( conn );

      if ( res == NULL ) {
         retval = dbi_memory_allocation_error;
         return NULL;
      }

      return new DBIRecordsetMySQL( m_dbh, res );
   }

   // query without recordset
   return NULL;
}

int DBITransactionMySQL::execute( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();

   if ( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 ) {
      switch ( mysql_errno( conn ) )
      {
      case CR_COMMANDS_OUT_OF_SYNC:
         retval = dbi_query_error;
         break;

      case CR_SERVER_GONE_ERROR:
      case CR_SERVER_LOST:
         retval = dbi_invalid_connection;
         break;

      default:
         retval = dbi_error;
      }
      return -1;
   }

   if ( mysql_field_count( conn ) > 0 ) {
      MYSQL_RES* res = mysql_store_result( conn );

      if ( res == NULL ) {
         retval = dbi_memory_allocation_error;
         return -1;
      }

      mysql_free_result( res );
   }

   retval = dbi_ok;

   // TODO: Convert this function to return an int64
   return (int) mysql_affected_rows( conn );
}

dbi_status DBITransactionMySQL::begin()
{
   dbi_status retval;

   execute( "BEGIN", retval );

   if ( retval == dbi_ok )
      m_inTransaction = true;

   return retval;
}

dbi_status DBITransactionMySQL::commit()
{
   dbi_status retval;

   execute( "COMMIT", retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionMySQL::rollback()
{
   dbi_status retval;

   execute( "ROLLBACK", retval );

   m_inTransaction = false;

   return retval;
}

void DBITransactionMySQL::close()
{
   // TODO: return a status code here because of the potential commit
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;

   m_dbh->closeTransaction( this );
}

dbi_status DBITransactionMySQL::getLastError( String &description )
{
   MYSQL *conn = static_cast<DBIHandleMySQL *>( m_dbh )->getConn();

   if ( conn == NULL )
      return dbi_invalid_connection;

   const char *errorMessage = mysql_error( conn );

   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}


DBIBlobStream *DBITransactionMySQL::openBlob( const String &blobId, dbi_status &status )
{
   status = dbi_not_implemented;
   return 0;
}

DBIBlobStream *DBITransactionMySQL::createBlob( dbi_status &status, const String &params,
      bool bBinary )
{
   status = dbi_not_implemented;
   return 0;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/
DBIHandleMySQL::~DBIHandleMySQL()
{
   DBIHandleMySQL::close();
}

DBITransaction *DBIHandleMySQL::startTransaction()
{
   DBITransactionMySQL *t = new DBITransactionMySQL( this );
   if ( t->begin() != dbi_ok ) {
      // TODO: filter useful information to the script level
      delete t;

      return NULL;
   }

   return t;
}

DBIHandleMySQL::DBIHandleMySQL()
{
   m_conn = NULL;
   m_connTr = NULL;
}

DBIHandleMySQL::DBIHandleMySQL( MYSQL *conn )
{
   m_conn = conn;
   m_connTr = NULL;
}

dbi_status DBIHandleMySQL::closeTransaction( DBITransaction *tr )
{
   return dbi_ok;
}

DBIRecordset *DBIHandleMySQL::query( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionMySQL( this );
   }

   return m_connTr->query( sql, retval );
}

int DBIHandleMySQL::execute( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionMySQL( this );
   }

   return m_connTr->execute( sql, retval );
}

int64 DBIHandleMySQL::getLastInsertedId()
{
   return mysql_insert_id( m_conn );
}

int64 DBIHandleMySQL::getLastInsertedId( const String& sequenceName )
{
   return mysql_insert_id( m_conn );
}

dbi_status DBIHandleMySQL::getLastError( String &description )
{
   if ( m_conn == NULL )
      return dbi_invalid_connection;

   const char *errorMessage = mysql_error( m_conn );
   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

dbi_status DBIHandleMySQL::escapeString( const String &value, String &escaped )
{
   if ( value.length() == 0 )
      return dbi_ok;

   AutoCString asValue( value );

   int maxLen = ( value.length() * 2 ) + 1;
   char *cTo = (char *) malloc( sizeof( char ) * maxLen );

   size_t convertedSize = mysql_real_escape_string( m_conn, cTo,
                                                   asValue.c_str(), asValue.length() );

   if ( convertedSize < value.length() ) {
      free( cTo );
      return dbi_error;
   }

   escaped = cTo;
   escaped.bufferize();

   free( cTo );

   return dbi_ok;
}

dbi_status DBIHandleMySQL::close()
{
   if ( m_conn != NULL ) {
      mysql_close( m_conn );
      m_conn = NULL;
   }

   return dbi_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

dbi_status DBIServiceMySQL::init()
{
   return dbi_ok;
}

DBIHandle *DBIServiceMySQL::connect( const String &parameters, bool persistent,
                                     dbi_status &retval, String &errorMessage )
{
   char *host, *user, *passwd, *db, *port, *unixSocket, *clientFlags;
   unsigned int iPort, iClientFlag;

   AutoCString asConnParams( parameters );
   char *connParams = (char *) malloc( sizeof(char) * (asConnParams.length() + 1) );
   strcpy( connParams, asConnParams.c_str() );


   host        = strtok( connParams, "," );
   user        = strtok( NULL, "," );
   passwd      = strtok( NULL, "," );
   db          = strtok( NULL, "," );
   port        = strtok( NULL, "," );
   unixSocket  = strtok( NULL, "," );
   clientFlags = strtok( NULL, "," );

   if ( strcmp( unixSocket, "0" ) == 0 )
      unixSocket = NULL;

   MYSQL *conn = mysql_init( NULL );

   if ( conn == NULL ) {
      retval = dbi_memory_allocation_error;
      free( connParams );
      return NULL;
   }

   if ( mysql_real_connect( conn, host, user, passwd, db, atoi( port ),
                           unixSocket, 0 ) == NULL )
   {
      errorMessage = mysql_error( conn );
      errorMessage.bufferize();
      mysql_close( conn );

      retval = dbi_connect_error;
      free( connParams );
      return NULL;
   }

   retval = dbi_ok;

   free( connParams );

   return new DBIHandleMySQL( conn );
}

CoreObject *DBIServiceMySQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "MySQL" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "MySQL" ) {
      vm->raiseModError( new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                                      .desc( "MySQL DBI driver was not found" ) ) );
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of mysql_srv.cpp */

