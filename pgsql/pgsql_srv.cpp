/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_srv.cpp
 *
 * PgSQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Sun Dec 23 21:54:42 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>

#include <falcon/engine.h>
#include "pgsql.h"

#define PG_TYPE_BOOL                    16
#define PG_TYPE_BYTEA                   17
#define PG_TYPE_CHAR                    18
#define PG_TYPE_NAME                    19
#define PG_TYPE_INT8                    20
#define PG_TYPE_INT2                    21
#define PG_TYPE_INT2VECTOR              22
#define PG_TYPE_INT4                    23
#define PG_TYPE_REGPROC                 24
#define PG_TYPE_TEXT                    25
#define PG_TYPE_OID                     26
#define PG_TYPE_TID                     27
#define PG_TYPE_XID                     28
#define PG_TYPE_CID                     29
#define PG_TYPE_OIDVECTOR               30
#define PG_TYPE_SET                     32
#define PG_TYPE_CHAR2                   409
#define PG_TYPE_CHAR4                   410
#define PG_TYPE_CHAR8                   411
#define PG_TYPE_POINT                   600
#define PG_TYPE_LSEG                    601
#define PG_TYPE_PATH                    602
#define PG_TYPE_BOX                     603
#define PG_TYPE_POLYGON                 604
#define PG_TYPE_FILENAME                605
#define PG_TYPE_FLOAT4                  700
#define PG_TYPE_FLOAT8                  701
#define PG_TYPE_ABSTIME                 702
#define PG_TYPE_RELTIME                 703
#define PG_TYPE_TINTERVAL               704
#define PG_TYPE_UNKNOWN                 705
#define PG_TYPE_MONEY                   790
#define PG_TYPE_OIDINT2                 810
#define PG_TYPE_OIDINT4                 910
#define PG_TYPE_OIDNAME                 911
#define PG_TYPE_BPCHAR                  1042
#define PG_TYPE_VARCHAR                 1043
#define PG_TYPE_DATE                    1082
#define PG_TYPE_TIME                    1083  /* w/o timezone */
#define PG_TYPE_TIMETZ                  1266  /* with timezone */
#define PG_TYPE_TIMESTAMP               1114  /* w/o timezone */
#define PG_TYPE_TIMESTAMPTZ             1184  /* with timezone */
#define PG_TYPE_NUMERIC                 1700

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetPgSQL::DBIRecordsetPgSQL( DBIHandle *dbh, PGresult *res )
    : DBIRecordset( dbh )
{
   m_res = res;

   m_row = -1; // BOF
   m_rowCount = PQntuples( m_res );
   m_columnCount = PQnfields( m_res );
}

DBIRecordsetPgSQL::~DBIRecordsetPgSQL()
{
   if ( m_res != NULL )
      close();
}

dbi_type DBIRecordsetPgSQL::getFalconType( Oid pgType )
{
   switch ( pgType )
   {
      case PG_TYPE_BOOL:
         return dbit_boolean;

      case PG_TYPE_INT2:
         return dbit_integer;

      case PG_TYPE_INT4: // TODO: are these right?
      case PG_TYPE_INT8:
         return dbit_integer64;

      case PG_TYPE_FLOAT4:
      case PG_TYPE_FLOAT8:
      case PG_TYPE_NUMERIC:
         return dbit_numeric;

      case PG_TYPE_DATE:
         return dbit_date;

      case PG_TYPE_TIME:
         return dbit_time;

      case PG_TYPE_TIMESTAMP:
         return dbit_datetime;

      default:
         return dbit_string;
   }
}

dbi_status DBIRecordsetPgSQL::next()
{
   m_row++;

   if ( m_row == m_rowCount )
      return dbi_eof;

   return dbi_ok;
}

int DBIRecordsetPgSQL::getColumnCount()
{
   return m_columnCount;
}

dbi_status DBIRecordsetPgSQL::getColumnNames( char *names[] )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      names[cIdx] = PQfname( m_res, cIdx );
   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::getColumnTypes( dbi_type *types )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
      types[cIdx] = getFalconType( PQftype( m_res, cIdx ) );

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asString( const int columnIndex, String &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );

   value = String( v );
   value.bufferize();

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asBoolean( const int columnIndex, bool &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );

   if ( strncmp( v, "t", 1 ) == 0 || strncmp( v, "T", 1 ) == 0 || strncmp( v, "1", 1 ) == 0 )
      value = true;
   else
      value = false;

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asInteger( const int columnIndex, int32 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );

   value = atoi( v );

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asInteger64( const int columnIndex, int64 &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );

   value = atoll( v );

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asNumeric( const int columnIndex, numeric &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );

   value = atof( v );

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asDate( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );
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

dbi_status DBIRecordsetPgSQL::asTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   String tv( v );

   // 01:02:03
   // 01234567

   int64 hour, minute, second;
   tv.subString( 0, 2 ).parseInt( hour );
   tv.subString( 3, 5 ).parseInt( minute );
   tv.subString( 6, 8 ).parseInt( second );

   Item zero( (int64) 0 );
   Item hr( hour );
   Item mn( minute );
   Item se( second );

   value.m_year = 0;
   value.m_month = 0;
   value.m_day = 0;
   value.m_hour = hour;
   value.m_minute = minute;
   value.m_second = second;
   value.m_msec = 0;

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asDateTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_columnCount )
      return dbi_column_range_error;
   else if ( m_res == NULL )
      return dbi_invalid_recordset;
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
      return dbi_nil_value;

   const char *v = PQgetvalue( m_res, m_row, columnIndex );
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

int DBIRecordsetPgSQL::getRowCount()
{
   return m_rowCount;
}

int DBIRecordsetPgSQL::getRowIndex()
{
   return m_row;
}

void DBIRecordsetPgSQL::close()
{
   if ( m_res != NULL ) {
      PQclear( m_res );
      m_res = NULL;
   }
}

dbi_status DBIRecordsetPgSQL::getLastError( String &description )
{
   PGconn *conn = ( (DBIHandlePgSQL *) m_dbh )->getPGconn();

   if ( conn == NULL )
      return dbi_invalid_connection;

   char *errorMessage = PQerrorMessage( conn );

   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

dbi_status DBIRecordsetPgSQL::asBlobID( const int columnIndex, String &value )
{
   return dbi_not_implemented;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionPgSQL::DBITransactionPgSQL( DBIHandle *dbh )
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionPgSQL::query( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   PGconn *conn = ((DBIHandlePgSQL *) m_dbh)->getPGconn();
   PGresult *res = PQexec( conn, asQuery.c_str() );

   if ( res == NULL ) {
      retval = dbi_memory_allocation_error;
      return NULL;
   }

   switch ( PQresultStatus( res ) )
   {
   case PGRES_TUPLES_OK:
      retval = dbi_ok;
      break;

   case PGRES_EMPTY_QUERY:
   case PGRES_COMMAND_OK:
   case PGRES_COPY_OUT:
   case PGRES_COPY_IN:
      retval = dbi_no_results;
      break;

   case PGRES_BAD_RESPONSE:
   case PGRES_NONFATAL_ERROR: // TODO: should this really trip an error?
   case PGRES_FATAL_ERROR:
      retval = dbi_execute_error;
      break;

   default:                  // Unknown error, this should never be reached
      retval = dbi_error;
      break;
   }

   if ( retval != dbi_ok )
      return NULL;

   return new DBIRecordsetPgSQL( m_dbh, res );
}

int DBITransactionPgSQL::execute( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   PGconn *conn = ((DBIHandlePgSQL *) m_dbh)->getPGconn();
   PGresult *res = PQexec( conn, asQuery.c_str() );

   if ( res == NULL ) {
      retval = dbi_memory_allocation_error;
      return 0;
   }

   int affectedRows;

   switch ( PQresultStatus( res ) )
   {
   case PGRES_EMPTY_QUERY:
   case PGRES_COMMAND_OK:
   case PGRES_TUPLES_OK:
   case PGRES_COPY_OUT:
   case PGRES_COPY_IN:
      {
         char *sAffectedRows = PQcmdTuples( res );
         if ( sAffectedRows == NULL || strlen( sAffectedRows ) == 0 )
            affectedRows = 0;
         else
            affectedRows = atoi( sAffectedRows );
         retval = dbi_ok;
      }
      break;

   case PGRES_BAD_RESPONSE:
   case PGRES_NONFATAL_ERROR: // TODO: should this really trip an error?
   case PGRES_FATAL_ERROR:
      retval = dbi_execute_error;
      affectedRows = -1;
      break;

   default:                  // Unknown error, this should never be reached
      retval = dbi_error;
      affectedRows = -1;
      break;
   }

   PQclear( res );

   return affectedRows;
}

dbi_status DBITransactionPgSQL::begin()
{
   dbi_status retval;

   execute( "BEGIN", retval );

   if ( retval == dbi_ok )
      m_inTransaction = true;

   return retval;
}

dbi_status DBITransactionPgSQL::commit()
{
   dbi_status retval;

   execute( "COMMIT", retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionPgSQL::rollback()
{
   dbi_status retval;

   execute( "ROLLBACK", retval );

   m_inTransaction = false;

   return retval;
}

void DBITransactionPgSQL::close()
{
   // TODO: return a status code here because of the potential commit
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;

   m_dbh->closeTransaction( this );
}

dbi_status DBITransactionPgSQL::getLastError( String &description )
{
   return dbi_ok;
}

DBIBlobStream *DBITransactionPgSQL::openBlob( const String &blobId, dbi_status &status )
{
   status = dbi_not_implemented;
   return 0;
}

DBIBlobStream *DBITransactionPgSQL::createBlob( dbi_status &status, const String &params,
      bool bBinary )
{
   status = dbi_not_implemented;
   return 0;
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandlePgSQL::~DBIHandlePgSQL()
{
   close();
}

DBITransaction *DBIHandlePgSQL::startTransaction()
{
   DBITransactionPgSQL *t = new DBITransactionPgSQL( this );
   if ( t->begin() != dbi_ok ) {
      // TODO: filter useful information to the script level
      delete t;

      return NULL;
   }

   return t;
}

DBIHandlePgSQL::DBIHandlePgSQL()
{
   m_conn = NULL;
   m_connTr = NULL;
}

DBIHandlePgSQL::DBIHandlePgSQL( PGconn *conn )
{
   m_conn = conn;
   m_connTr = NULL;
}

dbi_status DBIHandlePgSQL::closeTransaction( DBITransaction *tr )
{
   return dbi_ok;
}

DBIRecordset *DBIHandlePgSQL::query( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionPgSQL( this );
   }

   return m_connTr->query( sql, retval );
}

int DBIHandlePgSQL::execute( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionPgSQL( this );
   }

   return m_connTr->execute( sql, retval );
}

int64 DBIHandlePgSQL::getLastInsertedId()
{
   // PostgreSQL requires a sequence name
   return 0;
}

int64 DBIHandlePgSQL::getLastInsertedId( const String& sequenceName )
{
   char sql[128];
   AutoCString asSequenceName( sequenceName );

   snprintf( sql, 128, "SELECT CURRVAL('%s')", asSequenceName.c_str() );

   dbi_status retval;
   DBIRecordset *rs = query( sql, retval );

   int64 insertedId = 0;
   if ( retval == dbi_ok && rs->next() == 0 )
      rs->asInteger64( 0, insertedId );

   rs->close();

   return insertedId;
}


dbi_status DBIHandlePgSQL::getLastError( String &description )
{
   if ( m_conn == NULL )
      return dbi_invalid_connection;

   char *errorMessage = PQerrorMessage( m_conn );
   if ( errorMessage == NULL )
      return dbi_no_error_message;

   description.bufferize( errorMessage );

   return dbi_ok;
}

dbi_status DBIHandlePgSQL::escapeString( const String &value, String &escaped )
{
   if ( value.length() == 0 )
      return dbi_ok;

   AutoCString asValue( value );

   int maxLen = ( value.length() * 2 ) + 1;
   int errorCode;
   char *cTo = (char *) malloc( sizeof( char ) * maxLen );

   size_t convertedSize = PQescapeStringConn( m_conn, cTo, asValue.c_str(), maxLen,
                                              &errorCode );

   escaped = cTo;
   escaped.bufferize();

   free( cTo );

   return dbi_ok;
}

dbi_status DBIHandlePgSQL::close()
{
   if ( m_conn != NULL ) {
      PQfinish( m_conn );
      m_conn = NULL;
   }

   return dbi_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

dbi_status DBIServicePgSQL::init()
{
   return dbi_ok;
}

DBIHandle *DBIServicePgSQL::connect( const String &parameters, bool persistent,
                                     dbi_status &retval, String &errorMessage )
{
   AutoCString connParams( parameters );
   PGconn *conn = PQconnectdb( connParams.c_str () );
   if ( conn == NULL ) {
      retval = dbi_memory_allocation_error;
      return NULL;
   }

   if ( PQstatus( conn ) != CONNECTION_OK ) {
      retval = dbi_connect_error;
      errorMessage = PQerrorMessage( conn );
      errorMessage.remove( errorMessage.length() - 1, 1 ); // Get rid of newline
      errorMessage.bufferize();

      PQfinish( conn );

      return NULL;
   }

   retval = dbi_ok;

   return new DBIHandlePgSQL( conn );
}

CoreObject *DBIServicePgSQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "PgSQL" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "PgSQL" ) {
      vm->raiseModError( new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                                      .desc( "PgSQL DBI driver was not found" ) ) );
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of pgsql_srv.cpp */

