/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_mod.cpp

   SQLite3 driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:23:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "sqlite3_mod.h"

namespace Falcon
{


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetSQLite3::DBIRecordsetSQLite3( DBIHandle *dbh, sqlite3_stmt *res, bool bHasRow )
    : DBIRecordset( dbh ),
      m_bHasRow( bHasRow )
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
   int res;

   if ( m_bHasRow )
   {
      m_bHasRow = false;
      res = SQLITE_ROW;
   }
   else
      res = sqlite3_step( m_res );

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

   value.m_year = (int16) year;
   value.m_month = (int16) month;
   value.m_day = (int16) day;
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
   value.m_hour = (int16) hour;
   value.m_minute = (int16) minute;
   value.m_second = (int16) second;
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

   value.m_year = (int16) year;
   value.m_month = (int16) month;
   value.m_day = (int16) day;
   value.m_hour = (int16) hour;
   value.m_minute = (int16) minute;
   value.m_second = (int16) second;
   value.m_msec = 0;

   return dbi_ok;
}

dbi_status DBIRecordsetSQLite3::asBlobID( const int columnIndex, String &value )
{
   return dbi_not_implemented;
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
    : DBIStatement( dbh )
{
   m_inTransaction = false;
}


DBIRecordset* DBITransactionSQLite3::query( const String &query, int64 &affectedRows, dbi_status &retval )
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

   if ( status != SQLITE_OK ) {
      retval = dbi_error;
      affectedRows = -1;
      sqlite3_finalize( res );
      return 0;
   }


   status = sqlite3_step( res ); // execute the actual statement

   switch ( status )
   {
   case SQLITE_OK:
      // in case of SQLLITE_DONE, there is no recordset to be returned.
      sqlite3_finalize( res );
      affectedRows = 0;
      retval = dbi_ok;
      return 0;

   case SQLITE_DONE:
      affectedRows = sqlite3_changes( conn );
      sqlite3_finalize( res );
      retval = dbi_ok;
      return 0;

   case SQLITE_ROW:
      affectedRows = sqlite3_changes( conn );
      retval = dbi_ok;
      return new DBIRecordsetSQLite3( m_dbh, res, true );


   default: // TODO: provide better error info than this!
      retval = dbi_error;
      affectedRows = -1;
      break;
   }

   return 0;
}

dbi_status DBITransactionSQLite3::begin()
{
   dbi_status retval;
   int64 affected;
   query( "BEGIN", affected, retval );

   if ( retval == dbi_ok )
      m_inTransaction = true;

   return retval;
}

dbi_status DBITransactionSQLite3::commit()
{
   dbi_status retval;
   int64 affected;
   query( "COMMIT", affected, retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionSQLite3::rollback()
{
   dbi_status retval;
   int64 affected;
   query( "ROLLBACK", affected, retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionSQLite3::close()
{
   // TODO: return a status code here because of the potential commit
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;

   return m_dbh->closeTransaction( this );
}

dbi_status DBITransactionSQLite3::getLastError( String &description )
{
   sqlite3* conn = static_cast<DBIHandleSQLite3*>(getHandle())->getConn();

   const char *errorMessage = sqlite3_errmsg( conn );
   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;

}

DBIBlobStream *DBITransactionSQLite3::openBlob( const String &blobId, dbi_status &status )
{
   status = dbi_not_implemented;
   return 0;
}

DBIBlobStream *DBITransactionSQLite3::createBlob( dbi_status &status, const String &params,
      bool bBinary )
{
   status = dbi_not_implemented;
   return 0;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleSQLite3::DBIHandleSQLite3()
{
   m_conn = NULL;
}

DBIHandleSQLite3::DBIHandleSQLite3( sqlite3 *conn )
{
   m_conn = conn;
}

DBIHandleSQLite3::~DBIHandleSQLite3()
{
   close();
}

void DBIHandleSQLite3::options( const String& params )
{
   if( m_settings.parse( params ) )
   {
      // To turn off the autocommit.
      sqlite3_exec( m_conn, "BEGIN TRANSACTION", 0, 0, 0 );
   }
   else
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
   }
}

const DBIHandleSQLite3* DBIHandleMySQL::options() const
{
   return &m_settings;
}

DBIRecordset *DBIHandleSQLite3::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   AutoCString zSql( sql );
   sqlite3_stmt* pStmt;
   int res = sqlite3_prepare16_v2( m_conn, zSql.c_str(), zSql.length(), &pStmt, 0 );


}

void DBIHandleSQLite3::perform( const String &sql, int64 &affectedRows, const ItemArray& params )
{

}


DBIRecordset* DBIHandleSQLite3::call( const String &sql, int64 &affectedRows, const ItemArray& params )
{

}


DBIStatement* DBIHandleSQLite3::prepare( const String &query )
{

}


void DBIHandleSQLite3::begin()
{
   char* error;
   int res = sqlite3_exec( m_conn, "START TRANSACTION", 0, 0, &error );
   if( res != 0 )
      throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
}

void DBIHandleSQLite3::commit()
{
   char* error;
   int res = sqlite3_exec( m_conn, "COMMIT", 0, 0, &error );
   if( res != 0 )
      throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
}


void DBIHandleSQLite3::rollback()
{
   char* error;
   int res = sqlite3_exec( m_conn, "ROLLBACK", 0, 0, &error );
   if( res != 0 )
      throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
}


void DBIHandleSQLite3::selectLimited( const String& query,
      int64 nBegin, int64 nCount, String& result )
{
   String sBegin, sCount;

   if ( nBegin > 0 )
   {
      sBegin = " OFFSET ";
      sBegin.N( nBegin );
   }

   if( nCount > 0 )
   {
      sCount.N( nCount );
   }

   result = "SELECT " + query;

   if( nCount != 0 || nBegin != 0 )
   {
      result += "LIMIT " + sCount + sBegin;
   }
}


void DBIHandleSQLite3::throwError( int falconError, int sql3Error, const char* edesc )
{
   String err = String("(").N(sql3Error).A(")").A(edesc);

   throw new DBIError( ErrorParam(falconError, __LINE__ )
         .extra(err) );
}


int64 DBIHandleSQLite3::getLastInsertedId( const String& )
{
   return sqlite3_last_insert_rowid( m_conn );
}


void DBIHandleSQLite3::close()
{
   if ( m_conn != NULL )
   {
      sqlite3_exec( m_conn, "ROLLBACK", 0, 0, 0 );
      sqlite3_close( m_conn );
      m_conn = NULL;
   }

   return dbi_ok;
}

}

/* end of sqlite3_mod.cpp */

