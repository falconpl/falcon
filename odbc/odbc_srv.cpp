/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_srv.cpp
 *
 * MySQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Tiziano De Rubeis
 * Begin: Wed Oct 13 09:44:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <Windows.h>
#include <sqltypes.h>
#include <odbcss.h>
#include <falcon/engine.h>
#include "odbc_mod.h"
#include <sqlext.h>

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetODBC::DBIRecordsetODBC( DBIHandle *dbh, int nRows, int nCols )
    : DBIRecordset( dbh )
{
	m_nRow = -1;

	DBIHandleODBC* pConn = dynamic_cast<DBIHandleODBC*>( dbh );

	if( pConn == NULL )
	{
		m_sLastError = "Connection object is invalid!";
		m_nRowCount = -1;
		m_nColumnCount = -1;
		m_pDataArr = 0;
		return;
	}

	m_pConn = pConn->getConn( );
	m_nRowCount = nRows;
	m_nColumnCount = nCols;
	m_pDataArr = ( SRowData* )memAlloc( sizeof( SRowData ) * nCols );

	for ( int i = 0; i < nCols; ++i )
	{
		m_pDataArr[i].m_nLen = 0;
		m_pDataArr[i].m_pData = 0;
	}
}

DBIRecordsetODBC::~DBIRecordsetODBC()
{
	for ( int i = 0; i < m_nColumnCount; ++i )
		memFree( m_pDataArr[i].m_pData );

	memFree( m_pDataArr );
}

dbi_type DBIRecordsetODBC::getFalconType( int typ )
{
   switch ( typ )
   {
   case SQL_TINYINT:
   case SQL_INTEGER:
   case SQL_SMALLINT:
   case SQL_BIT:
      return dbit_integer;

   case SQL_BIGINT:
      return dbit_integer64;

   case SQL_DECIMAL:
   case SQL_NUMERIC:
   case SQL_FLOAT:
   case SQL_REAL:
   case SQL_DOUBLE:
      return dbit_numeric;

   case SQL_TYPE_DATE:
      return dbit_date;

   case SQL_TYPE_TIME:
      return dbit_time;

   case SQL_TYPE_TIMESTAMP:
      return dbit_datetime;

   case SQL_BINARY:
   case SQL_VARBINARY:
   case SQL_LONGVARBINARY:
	  return dbit_blob;

   // In this version interval data type is not supported
   default:
      return dbit_string;
   }
}

dbi_status DBIRecordsetODBC::next()
{
	SQLRETURN Ret = SQLFetch( m_pConn->m_hHstmt );

	if( Ret != SQL_SUCCESS && Ret != SQL_SUCCESS_WITH_INFO && Ret != SQL_NO_DATA )
	{
		m_sLastError = GetErrorMessage( SQL_HANDLE_STMT, m_pConn->m_hHstmt, TRUE );
		return dbi_error;
	}

	if( Ret == SQL_NO_DATA )
		return dbi_eof;

	m_nRow++;
	return dbi_ok;
}

int DBIRecordsetODBC::getColumnCount()
{
   return m_nColumnCount;
}

dbi_status DBIRecordsetODBC::getColumnNames( char *names[] )
{
	RETCODE ret;
	SQLCHAR ColName[1024];
	SQLSMALLINT nLen, nType, nDec, nNull;
	SQLUINTEGER nColLen;

	for ( int cIdx = 1; cIdx <= m_nColumnCount; cIdx++ )
	{
		ret = SQLDescribeCol( m_pConn->m_hHstmt, cIdx, ColName, 1024, &nLen, &nType, &nColLen, &nDec, &nNull );

		if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
		{
			m_sLastError = GetErrorMessage( SQL_HANDLE_STMT, m_pConn->m_hHstmt, TRUE );
			return dbi_error;
		}

		ColName[nLen] = 0;
		strcpy( names[cIdx - 1], ( char* )ColName );
	}

	return dbi_ok;
}

dbi_status DBIRecordsetODBC::getColumnTypes( dbi_type *types )
{
	RETCODE ret;
	SQLCHAR ColName[1024];
	SQLSMALLINT nLen, nType, nDec, nNull;
	SQLUINTEGER nColLen;

	for ( int cIdx = 1; cIdx <= m_nColumnCount; cIdx++ )
	{
		ret = SQLDescribeCol( m_pConn->m_hHstmt, cIdx, ColName, 1024, &nLen, &nType, &nColLen, &nDec, &nNull );

		if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
		{
			m_sLastError = GetErrorMessage( SQL_HANDLE_STMT, m_pConn->m_hHstmt, TRUE );
			return dbi_error;
		}

		types[cIdx - 1] = getFalconType( nType );
	}

	return dbi_ok;
}

dbi_status DBIRecordsetODBC::asString( const int columnIndex, String &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   // TODO: check proper field encoding and transcode.
   value = String( ( char* )( m_pDataArr[columnIndex].m_pData ), m_pDataArr[columnIndex].m_nLen );
   return dbi_ok;
}

dbi_status DBIRecordsetODBC::asBlobID( const int columnIndex, String &value )
{
   return dbi_not_implemented;
}

dbi_status DBIRecordsetODBC::asBoolean( const int columnIndex, bool &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   char *v = ( char* )( m_pDataArr[columnIndex].m_pData );

   if (strncmp( v, "t", 1 ) == 0 || strncmp( v, "T", 1 ) == 0 || strncmp( v, "1", 1 ) == 0)
      value = true;
   else
      value = false;

   return dbi_ok;
}

dbi_status DBIRecordsetODBC::asInteger( const int columnIndex, int32 &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   ( ( char* )( m_pDataArr[columnIndex].m_pData ) )[m_pDataArr[columnIndex].m_nLen] = 0;
   value = atoi( ( char* )( m_pDataArr[columnIndex].m_pData ) );

   return dbi_ok;
}

dbi_status DBIRecordsetODBC::asInteger64( const int columnIndex, int64 &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   ( ( char* )( m_pDataArr[columnIndex].m_pData ) )[m_pDataArr[columnIndex].m_nLen] = 0;
   value = atoll( ( char* )( m_pDataArr[columnIndex].m_pData ) );

   return dbi_ok;
}

dbi_status DBIRecordsetODBC::asNumeric( const int columnIndex, numeric &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   ( ( char* )( m_pDataArr[columnIndex].m_pData ) )[m_pDataArr[columnIndex].m_nLen] = 0;
   value = atof( ( char* )( m_pDataArr[columnIndex].m_pData ) );

   return dbi_ok;
}

dbi_status DBIRecordsetODBC::asDate( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   String tv( ( char* )( m_pDataArr[columnIndex].m_pData ), m_pDataArr[columnIndex].m_nLen );

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

dbi_status DBIRecordsetODBC::asTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   String tv( ( char* )( m_pDataArr[columnIndex].m_pData ), m_pDataArr[columnIndex].m_nLen );

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

dbi_status DBIRecordsetODBC::asDateTime( const int columnIndex, TimeStamp &value )
{
   if ( columnIndex >= m_nColumnCount )
      return dbi_column_range_error;
   else if ( m_pConn == NULL )
      return dbi_invalid_recordset;
   else if ( m_pDataArr[columnIndex].m_pData == NULL )
      return dbi_nil_value;

   String tv( ( char* )( m_pDataArr[columnIndex].m_pData ), m_pDataArr[columnIndex].m_nLen );

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

int DBIRecordsetODBC::getRowCount()
{
   return m_nRowCount;
}

int DBIRecordsetODBC::getRowIndex()
{
   return m_nRow;
}

void DBIRecordsetODBC::close()
{
	for ( int i = 0; i < m_nColumnCount; ++i )
		memFree( m_pDataArr[i].m_pData );

	memFree( m_pDataArr );
}

dbi_status DBIRecordsetODBC::getLastError( String &description )
{
   description = m_sLastError;
   return dbi_ok;
}

dbi_status DBIRecordsetODBC::bind( int ord, int type )
{
	if( m_nColumnCount < ord )
	{
		m_pDataArr = ( SRowData* )memRealloc( m_pDataArr, ord * sizeof( SRowData ) );
		m_nColumnCount = ord;
	}

	if( m_pDataArr[ord - 1].m_pData )
		memFree( m_pDataArr[ord - 1].m_pData );

	m_pDataArr[ord - 1].m_pData = memAlloc( 0x2000 );

	RETCODE ret = SQLBindCol( m_pConn->m_hHstmt, ord, type, m_pDataArr[ord - 1].m_pData, 0x2000, ( SQLINTEGER* )( &m_pDataArr[ord - 1].m_nLen ) );

	if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
	{
		m_sLastError = GetErrorMessage( SQL_HANDLE_STMT, m_pConn->m_hHstmt, TRUE );
		return dbi_query_error;
	}

	return dbi_ok;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionODBC::DBITransactionODBC( DBIHandle *dbh )
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionODBC::query( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   ODBCConn *conn = ((DBIHandleODBC *) m_dbh)->getConn();

   RETCODE ret = SQLExecDirect( conn->m_hHstmt, ( SQLCHAR* )asQuery.c_str( ), asQuery.length() );

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      retval = dbi_query_error;
      return NULL;
   }

   SQLINTEGER nRowCount;
   RETCODE retcode = SQLRowCount( conn->m_hHstmt, &nRowCount );

   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {
	   retval = dbi_query_error;
	   return NULL;
   }

   retval = dbi_ok;

   if( nRowCount != 0 )
   {
	   SQLSMALLINT nColCount;
	   retcode = SQLNumResultCols( conn->m_hHstmt, &nColCount );

      if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
	   {
		   retval = dbi_query_error;
		   return NULL;
	   }

	   return new DBIRecordsetODBC( m_dbh, nRowCount, nColCount );
   }

   // query without recordset
   return NULL;
}

int DBITransactionODBC::execute( const String &query, dbi_status &retval )
{
   AutoCString asQuery( query );
   ODBCConn *conn = ((DBIHandleODBC *) m_dbh)->getConn();

   RETCODE ret = SQLExecDirect( conn->m_hHstmt, ( SQLCHAR* )asQuery.c_str( ), SQL_NTS );

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
	   retval = dbi_query_error;
	   return -1;
   }

   SQLINTEGER nRowCount;
   ret = SQLRowCount( conn->m_hHstmt, &nRowCount );

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
	   retval = dbi_query_error;
	   return -1;
   }

   retval = dbi_ok;
   return nRowCount;
}

dbi_status DBITransactionODBC::begin()
{
   dbi_status retval;

   execute( "BEGIN", retval );

   if ( retval == dbi_ok )
      m_inTransaction = true;

   return retval;
}

dbi_status DBITransactionODBC::commit()
{
   dbi_status retval;

   execute( "COMMIT", retval );

   m_inTransaction = false;

   return retval;
}

dbi_status DBITransactionODBC::rollback()
{
   dbi_status retval;

   execute( "ROLLBACK", retval );

   m_inTransaction = false;

   return retval;
}

void DBITransactionODBC::close()
{
   // TODO: return a status code here because of the potential commit
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;

   m_dbh->closeTransaction( this );
}

dbi_status DBITransactionODBC::getLastError( String &description )
{
	description = m_sLastError;
	return dbi_ok;
}


DBIBlobStream *DBITransactionODBC::openBlob( const String &blobId, dbi_status &status )
{
   status = dbi_not_implemented;
   return 0;
}

DBIBlobStream *DBITransactionODBC::createBlob( dbi_status &status, const String &params,
      bool bBinary )
{
   status = dbi_not_implemented;
   return 0;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBITransaction *DBIHandleODBC::startTransaction()
{
   DBITransactionODBC *t = new DBITransactionODBC( this );
   if ( t->begin() != dbi_ok ) {
      // TODO: filter useful information to the script level
      delete t;

      return NULL;
   }

   return t;
}

DBIHandleODBC::DBIHandleODBC()
{
	m_conn = NULL;
	m_connTr = NULL;
}

DBIHandleODBC::DBIHandleODBC( ODBCConn *conn )
{
   m_conn = conn;
   m_connTr = NULL;
}

DBIHandleODBC::~DBIHandleODBC( )
{
	close( );
}

dbi_status DBIHandleODBC::closeTransaction( DBITransaction *tr )
{
   return dbi_ok;
}

DBIRecordset *DBIHandleODBC::query( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionODBC( this );
   }

   return m_connTr->query( sql, retval );
}

int DBIHandleODBC::execute( const String &sql, dbi_status &retval )
{
   if ( m_connTr == NULL ) {
      m_connTr = new DBITransactionODBC( this );
   }

   return m_connTr->execute( sql, retval );
}

int64 DBIHandleODBC::getLastInsertedId()
{
   return -1;
}

int64 DBIHandleODBC::getLastInsertedId( const String& sequenceName )
{
   return -1;
}

dbi_status DBIHandleODBC::getLastError( String &description )
{
   GetErrorMessage( SQL_HANDLE_STMT, m_conn->m_hHstmt, TRUE );

   return dbi_ok;
}

dbi_status DBIHandleODBC::escapeString( const String &value, String &escaped )
{
	if ( value.length() == 0 )
	  return dbi_ok;

	if( m_conn == NULL )
	   return dbi_invalid_connection;

	AutoCString sConv( value );
	int maxLen = ( sConv.length() * 2 ) + 1;
	SQLCHAR* pRet = (SQLCHAR *) malloc( sizeof( SQLCHAR ) * maxLen );
	SQLINTEGER nBuff, nBuffOut;

	RETCODE ret = SQLNativeSql( m_conn->m_hHdbc, ( SQLCHAR* )sConv.c_str( ), sConv.length( ), pRet, nBuff, &nBuffOut );

	if( ( ret != SQL_SUCCESS ) && ( ret != SQL_SUCCESS_WITH_INFO ) )
		return dbi_execute_error;

   escaped = ( char* )pRet;
   escaped.bufferize();

   free( pRet );

   return dbi_ok;
}

dbi_status DBIHandleODBC::close()
{
	if( m_conn )
	{
		m_conn->Destroy( );
		memFree( m_conn );
		m_conn = NULL;
	}
	
	return dbi_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

dbi_status DBIServiceODBC::init()
{
   return dbi_ok;
}

DBIHandle *DBIServiceODBC::connect( const String &parameters, bool persistent,
                                     dbi_status &retval, String &errorMessage )
{
   AutoCString asConnParams( parameters );
   char *connParams = (char *) memAlloc( sizeof(char) * (asConnParams.length() + 1) );
   strcpy( connParams, asConnParams.c_str() );

   SQLHDESC hIpd;
   SQLHENV hEnv;
   SQLHDBC hHdbc;
   SQLHSTMT hHstmt;

   RETCODE retcode = SQLAllocHandle (SQL_HANDLE_ENV, NULL, &hEnv);

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
	   retval = dbi_connect_error;
	   errorMessage = "Impossible to allocate the ODBC environment";
	   memFree( connParams );
	   return NULL;
   }

   retcode = SQLSetEnvAttr( hEnv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER) SQL_OV_ODBC3, SQL_IS_INTEGER );

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   retval = dbi_connect_error;
	   errorMessage = "Impossible to notify ODBC that this is an ODBC 3.0 app.";
	   memFree( connParams );
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   return NULL;
   }

   // Allocate ODBC connection handle and connect.
   retcode = SQLAllocHandle( SQL_HANDLE_DBC, hEnv, &hHdbc );

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   retval = dbi_connect_error;
	   errorMessage = "Impossible to allocate ODBC connection handle and connect.";
	   memFree( connParams );
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   return NULL;
   }

   int nSec = 30;
   SQLSetConnectAttr( hHdbc, SQL_LOGIN_TIMEOUT, (SQLPOINTER)(&nSec), 0 );

   SQLCHAR OutConnStr[MAXBUFLEN];
   short OutConnStrLen = MAXBUFLEN;

   retcode = SQLDriverConnect(
	   hHdbc, 
	   NULL, 
//	   (SQLCHAR*)sConn.c_str(),
	   (SQLCHAR*)connParams,
	   strlen(connParams),
	   OutConnStr,
	   MAXBUFLEN, 
	   &OutConnStrLen,
	   SQL_DRIVER_NOPROMPT );

   if( ( retcode != SQL_SUCCESS ) && ( retcode != SQL_SUCCESS_WITH_INFO ) )
   {
	   errorMessage = "SQLDriverConnect failed. Reason: " + GetErrorMessage( SQL_HANDLE_DBC, hHdbc, FALSE );
	   memFree( connParams );
	   SQLDisconnect( hHdbc );
	   SQLFreeHandle( SQL_HANDLE_DBC, hHdbc );
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   return NULL;
   }

   retcode = SQLAllocHandle( SQL_HANDLE_STMT, hHdbc, &hHstmt );

   if( ( retcode != SQL_SUCCESS ) && ( retcode != SQL_SUCCESS_WITH_INFO ) )
   {
	   errorMessage = "SQLAllocHandle failed. Reason: " + GetErrorMessage( SQL_HANDLE_DBC, hHdbc, TRUE );
	   memFree( connParams );
	   SQLDisconnect( hHdbc );
	   SQLFreeHandle( SQL_HANDLE_DBC, hHdbc );
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   return NULL;
   }

   retcode = SQLGetStmtAttr( hHstmt, SQL_ATTR_IMP_PARAM_DESC, &hIpd, 0, 0 );

   if( (retcode != SQL_SUCCESS) && (retcode != SQL_SUCCESS_WITH_INFO) )
   {
	   errorMessage = "SQLGetStmtAttr failed. Reason: " + GetErrorMessage( SQL_HANDLE_STMT, hHstmt, TRUE );
	   memFree( connParams );
	   SQLFreeHandle( SQL_HANDLE_STMT, hHstmt );
	   SQLDisconnect( hHdbc );
	   SQLFreeHandle( SQL_HANDLE_DBC, hHdbc );
	   SQLFreeHandle(SQL_HANDLE_ENV, hEnv );
	   return NULL;
   }

   memFree( connParams );
   ODBCConn* conn = ( ODBCConn* )memAlloc( sizeof( ODBCConn ) );
   conn->Initialize( hEnv, hHdbc, hHstmt, hIpd );

   retval = dbi_ok;
   return new DBIHandleODBC( conn );
}

CoreObject *DBIServiceODBC::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "ODBC" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "ODBC" ) {
      vm->raiseModError( new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                                      .desc( "ODBC DBI driver was not found" ) ) );
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

String GetErrorMessage(SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd)
{
	RETCODE      plm_retcode = SQL_SUCCESS;
	UCHAR      plm_szSqlState[MAXBUFLEN] = "",
		plm_szErrorMsg[MAXBUFLEN] = "";
	SDWORD      plm_pfNativeError = 0L;
	SWORD      plm_pcbErrorMsg = 0;
	SQLSMALLINT   plm_cRecNmbr = 1;
	SDWORD      plm_SS_MsgState = 0, plm_SS_Severity = 0;
	SQLINTEGER   plm_Rownumber = 0;
	USHORT      plm_SS_Line;
	SQLSMALLINT   plm_cbSS_Procname, plm_cbSS_Srvname;
	SQLCHAR      plm_SS_Procname[MAXNAME], plm_SS_Srvname[MAXNAME];
	String sRet = "";
	char Convert[MAXBUFLEN];

	while (plm_retcode != SQL_NO_DATA_FOUND) {
		plm_retcode = SQLGetDiagRec(plm_handle_type, plm_handle,
			plm_cRecNmbr, plm_szSqlState, &plm_pfNativeError,
			plm_szErrorMsg, MAXBUFLEN - 1, &plm_pcbErrorMsg);

		// Note that if the application has not yet made a
		// successful connection, the SQLGetDiagField
		// information has not yet been cached by ODBC
		// Driver Manager and these calls to SQLGetDiagField
		// will fail.
		if (plm_retcode != SQL_NO_DATA_FOUND) {
			if (ConnInd) {
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_ROW_NUMBER, &plm_Rownumber,
					SQL_IS_INTEGER,
					NULL);
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_SS_LINE, &plm_SS_Line,
					SQL_IS_INTEGER,
					NULL);
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_SS_MSGSTATE, &plm_SS_MsgState,
					SQL_IS_INTEGER,
					NULL);
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_SS_SEVERITY, &plm_SS_Severity,
					SQL_IS_INTEGER,
					NULL);
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_SS_PROCNAME, &plm_SS_Procname,
					sizeof(plm_SS_Procname),
					&plm_cbSS_Procname);
				plm_retcode = SQLGetDiagField(
					plm_handle_type, plm_handle, plm_cRecNmbr,
					SQL_DIAG_SS_SRVNAME, &plm_SS_Srvname,
					sizeof(plm_SS_Srvname),
					&plm_cbSS_Srvname);
			}

			sRet += "SqlState = " + String( ( char* )plm_szSqlState ) + ";";
			sRet += "NativeError = " + String( _itoa( plm_pfNativeError, Convert, 10 ) ) + ";";
			sRet += "ErrorMsg = " + String( ( char* )plm_szErrorMsg ) + ";";
			sRet += "pcbErrorMsg = " + String( _itoa( plm_pcbErrorMsg, Convert, 10 ) ) + ";";

			if (ConnInd)
			{
				sRet += "ODBCRowNumber = " + String( _itoa( plm_Rownumber, Convert, 10 ) ) + ";";
				sRet += "SSrvrLine = " + String( _itoa( plm_Rownumber, Convert, 10 ) ) + ";";
				sRet += "SSrvrMsgState = " + String( _itoa( plm_SS_MsgState, Convert, 10 ) ) + ";";
				sRet += "SSrvrSeverity = " + String( _itoa( plm_SS_Severity, Convert, 10 ) ) + ";";
				sRet += "SSrvrProcname = " + String( ( char* )plm_SS_Procname ) + ";";
				sRet += "SSrvrSrvname = " + String( ( char* )plm_SS_Srvname ) + ";";
			}
		}

		plm_cRecNmbr++; //Increment to next diagnostic record.
	}

	return sRet;
}

} /* namespace Falcon */

/* end of mysql_srv.cpp */

