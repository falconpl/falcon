/*
   FALCON - The Falcon Programming Language.
   FILE: pgsql_srv.cpp
   
   PgSQL Falcon service/driver
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: Sun Dec 23 21:54:42 2007
   Last modified because:
   
   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
   
   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
 */

#include <string.h>

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
   {
      close();
   }
}

dbi_type DBIRecordsetPgSQL::getFalconType( Oid pgType )
{
   switch ( pgType )
   {
   case PG_TYPE_BOOL:
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

DBIRecordset::dbr_status DBIRecordsetPgSQL::next()
{
   m_row++;
   
   if ( m_row == m_rowCount )
   {
      return s_eof;
   }
   
   return s_ok;
}

int DBIRecordsetPgSQL::getColumnCount()
{
   return m_columnCount;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::getColumnNames( CoreArray *resultCache )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
   {
      char *fname = PQfname( m_res, cIdx );
      GarbageString *gsFName = new GarbageString( resultCache->origin() );
      gsFName->bufferize( fname );
      resultCache->append( gsFName );
   }
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::getColumnTypes( CoreArray *resultCache )
{
   for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
   {
      dbi_type typ = getFalconType( PQftype( m_res, cIdx ) );
      resultCache->append( (int64) typ );
   }
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asString( const int columnIndex, String &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   
   value = String( v );
   value.bufferize();
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asInteger( const int columnIndex, int32 &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   
   value = atoi( v );
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asInteger64( const int columnIndex, int64 &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   
   // TODO: is this conversion correct?
   value = atoll( v );
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asNumeric( const int columnIndex, numeric &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   
   value = atof( v );
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asDate( const int columnIndex, CoreObject &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   String tv( v );
   
   // 2007-12-27
   // 0123456789
   
   int64 year, month, day;
   tv.subString( 0, 4 ).parseInt( year );
   tv.subString( 5, 7 ).parseInt( month );
   tv.subString( 8, 10 ).parseInt( day );
   
   value.setProperty( "year",  year );
   value.setProperty( "month", month );
   value.setProperty( "day",   day );
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asTime( const int columnIndex, CoreObject &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
   const char *v = PQgetvalue( m_res, m_row, columnIndex );
   String tv( v );
   
   // 01:02:03
   // 01234567
   
   int64 hour, minute, second;
   tv.subString( 0, 2 ).parseInt( hour );
   tv.subString( 3, 5 ).parseInt( minute );
   tv.subString( 6, 8 ).parseInt( second );
   
   value.setProperty( "hour",   hour );
   value.setProperty( "minute", minute );
   value.setProperty( "second", second );
   
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::asDateTime( const int columnIndex, CoreObject &value )
{
   if ( columnIndex >= m_columnCount )
   {
      return s_column_range_error;
   }
   else if ( m_res == NULL )
   {
      return s_invalid_record_handle;
   }
   else if ( PQgetisnull( m_res, m_row, columnIndex ) == 1 )
   {
      return s_nil_value;
   }
   
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
   
   value.setProperty( "year",  year );
   value.setProperty( "month", month );
   value.setProperty( "day",   day );
   value.setProperty( "hour",   hour );
   value.setProperty( "minute", minute );
   value.setProperty( "second", second );
   
   return s_ok;
}

int DBIRecordsetPgSQL::getRowCount()
{
   return m_rowCount;
}

void DBIRecordsetPgSQL::close()
{
   if ( m_res != NULL )
   {
      PQclear( m_res );
      m_res = NULL;
   }
}

DBIRecordsetPgSQL::dbr_status DBIRecordsetPgSQL::getLastError( String &description )
{
   return s_ok;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionPgSQL::DBITransactionPgSQL( DBIHandle *dbh ) 
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionPgSQL::query( const String &query, dbt_status &retval )
{
   AutoCString asQuery( query );
   PGconn *conn = ((DBIHandlePgSQL *) m_dbh)->getPGconn();
   PGresult *res = PQexec( conn, asQuery.c_str() );
   
   if ( res == NULL )
   {
      retval = s_memory_allocation_error;
      return NULL;
   }
   
   switch ( PQresultStatus( res ) )
   {
   case PGRES_TUPLES_OK:
      retval = s_ok;
      break;
      
   case PGRES_EMPTY_QUERY:
   case PGRES_COMMAND_OK:
   case PGRES_COPY_OUT:
   case PGRES_COPY_IN:
      retval = s_no_results;
      break;
      
   case PGRES_BAD_RESPONSE:
   case PGRES_NONFATAL_ERROR: // TODO: should this really trip an error?
   case PGRES_FATAL_ERROR:
      retval = s_execute_error;
      break;
      
   default:                  // Unknown error, this should never be reached
      retval = s_error;
      break;
   }
   
   if ( retval != s_ok )
   {
      return NULL;
   }
   
   return new DBIRecordsetPgSQL( m_dbh, res );
}

int DBITransactionPgSQL::execute( const String &query, dbt_status &retval )
{
   AutoCString asQuery( query );
   PGconn *conn = ((DBIHandlePgSQL *) m_dbh)->getPGconn();
   PGresult *res = PQexec( conn, asQuery.c_str() );
   
   if ( res == NULL )
   {
      retval = s_memory_allocation_error;
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
         {
            affectedRows = 0;
         }
         else
         {
            affectedRows = atoi( sAffectedRows );
         }
         retval = s_ok;
      }
      break;
      
   case PGRES_BAD_RESPONSE:
   case PGRES_NONFATAL_ERROR: // TODO: should this really trip an error?
   case PGRES_FATAL_ERROR:
      retval = s_execute_error;
      affectedRows = -1;
      break;
      
   default:                  // Unknown error, this should never be reached
      retval = s_error;
      affectedRows = -1;
      break;
   }
   
   PQclear( res );
   
   return affectedRows;
}

DBITransaction::dbt_status DBITransactionPgSQL::begin()
{
   dbt_status retval;
   
   execute( "BEGIN", retval );
   
   if ( retval == s_ok )
   {
      m_inTransaction = true;
   }
   
   return retval;
}

DBITransaction::dbt_status DBITransactionPgSQL::commit()
{
   dbt_status retval;
   
   execute( "COMMIT", retval );
   
   m_inTransaction = false;
   
   return retval;
}

DBITransaction::dbt_status DBITransactionPgSQL::rollback()
{
   dbt_status retval;
   
   execute( "ROLLBACK", retval );
   
   m_inTransaction = false;
   
   return retval;
}

void DBITransactionPgSQL::close()
{
   if ( m_inTransaction )
   {
      commit();
   }
   
   m_inTransaction = false;
   
   m_dbh->closeTransaction( this );
}

DBITransaction::dbt_status DBITransactionPgSQL::getLastError( String &description )
{
   return s_ok;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBITransaction *DBIHandlePgSQL::startTransaction()
{
   DBITransactionPgSQL *t = new DBITransactionPgSQL( this );
   if ( t->begin() != DBITransaction::s_ok )
   {
      // TODO: set error state
      
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

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::closeTransaction( DBITransaction *tr )
{
   return s_ok;
}

DBIRecordset *DBIHandlePgSQL::query( const String &sql, DBITransaction::dbt_status &retval )
{
   if ( m_connTr == NULL )
   {
      m_connTr = new DBITransactionPgSQL( this );
   }
   
   return m_connTr->query( sql, retval );
}

int DBIHandlePgSQL::execute( const String &sql, DBITransaction::dbt_status &retval )
{
   if ( m_connTr == NULL )
   {
      m_connTr = new DBITransactionPgSQL( this );
   }
   
   return m_connTr->execute( sql, retval );
}

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::getLastError( String &description )
{
   return s_ok;
}

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::close()
{
   if ( m_conn != NULL )
   {
      PQfinish( m_conn );
      m_conn = NULL;
   }
   
   return s_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

DBIServicePgSQL::dbi_status DBIServicePgSQL::init()
{
   return s_ok;
}

DBIHandle *DBIServicePgSQL::connect( const String &parameters, bool persistent, 
                                     dbi_status &retval, String &errorMessage )
{
   AutoCString connParams( parameters );
   PGconn *conn = PQconnectdb( connParams.c_str () );
   if ( conn == NULL ) {
      retval = s_memory_alloc_error;
      return NULL;
   }
   
   if ( PQstatus( conn ) != CONNECTION_OK ) {
      retval = s_connect_failed;
      // TODO: Use append? I used append because copy and = were causing memory
      // errors because the memory PQerrorMessage is pointing to is free'd when
      // the later PQfinish is called. I would have thought that .copy() would
      // have taken care of this, but I suffered the same corrupt string
      // symptoms with copy as I did =. = with a strdup worked fine, but I was
      // then worried about memory leaks.
      errorMessage.append( PQerrorMessage( conn ) );
      errorMessage.remove( errorMessage.length() - 1, 1 ); // Get rid of newline
      
      PQfinish( conn );
      
      return NULL;
   }
   
   retval = s_ok;
   
   return new DBIHandlePgSQL( conn );
}

CoreObject *DBIServicePgSQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findGlobalItem( "PgSQL" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "PgSQL" )
   {
      // TODO: raise an error.
      return 0;
   }
   
   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );
   
   return obj;
}

} /* namespace Falcon */

/* end of pgsql_srv.cpp */

