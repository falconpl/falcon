/*
   FALCON - The Falcon Programming Language.
   FILE: odbc_mod.cpp

   ODBC driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:23:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "odbc_mod.h"
#include <string.h>

namespace Falcon
{

/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

Sqlite3InBind::Sqlite3InBind( odbc_stmt* stmt ):
      DBIInBind(true),  // always changes binding
      m_stmt(stmt)
{}

Sqlite3InBind::~Sqlite3InBind()
{
   // nothing to do: the statement is not ours.
}


void Sqlite3InBind::onFirstBinding( int size )
{
   // nothing to allocate here.
}

void Sqlite3InBind::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];

   switch( item.type() )
   {
   // set to null
   case DBIBindItem::t_nil:
      odbc_bind_null( m_stmt, num+1 );
      break;

   case DBIBindItem::t_bool:
   case DBIBindItem::t_int:
      odbc_bind_int64( m_stmt, num+1, item.asInteger() );
      break;

   case DBIBindItem::t_double:
      odbc_bind_double( m_stmt, num+1, item.asDouble() );
      break;

   case DBIBindItem::t_string:
      odbc_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_STATIC );
      break;

   case DBIBindItem::t_buffer:
      odbc_bind_blob( m_stmt, num+1, item.asBuffer(), item.asStringLen(), SQLITE_STATIC );
      break;

   // the time has normally been decoded in the buffer
   case DBIBindItem::t_time:
      odbc_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_STATIC );
      break;
   }
}


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetODBC::DBIRecordsetODBC( DBIHandleODBC *dbh, odbc_stmt *res, const ItemArray& params )
    : DBIRecordset( dbh ),
      m_stmt( res ),
      m_bind( res )
{
   m_bAsString = dbh->options()->m_bFetchStrings;
   m_bind.bind( params );
   m_row = -1; // BOF
   m_columnCount = odbc_column_count( res );
}

DBIRecordsetODBC::~DBIRecordsetODBC()
{
   if ( m_stmt != NULL )
      close();
}

int DBIRecordsetODBC::getColumnCount()
{
   return m_columnCount;
}

int64 DBIRecordsetODBC::getRowIndex()
{
   return m_row;
}

int64 DBIRecordsetODBC::getRowCount()
{
   return -1; // we don't know
}


bool DBIRecordsetODBC::getColumnName( int nCol, String& name )
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   name.bufferize( odbc_column_name( m_stmt, nCol ) );

   return true;
}


bool DBIRecordsetODBC::fetchRow()
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   int res = odbc_step( m_stmt );

   if( res == SQLITE_DONE )
      return false;
   else if ( res != SQLITE_ROW )
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_FETCH, res );

   // more data incoming
   m_row++;
   return true;
}


bool DBIRecordsetODBC::discard( int64 ncount )
{
   while ( ncount > 0 )
   {
      if( ! fetchRow() )
      {
         return false;
      }
      --ncount;
   }

   return true;
}


void DBIRecordsetODBC::close()
{
   if( m_stmt != 0 )
   {
      odbc_finalize( m_stmt );
      m_stmt = 0;
   }
}

bool DBIRecordsetODBC::getColumnValue( int nCol, Item& value )
{
   if( m_stmt == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   switch ( odbc_column_type(m_stmt, nCol) )
   {
   case SQLITE_NULL:
      value.setNil();
      return true;

   case SQLITE_INTEGER:
      if( m_bAsString )
      {
         value = new CoreString( (const char*)odbc_column_text(m_stmt, nCol), -1 );
      }
      else
      {
         value.setInteger( odbc_column_int64(m_stmt, nCol) );
      }
      return true;

   case SQLITE_FLOAT:
      if( m_bAsString )
      {
         value = new CoreString( (const char*)odbc_column_text( m_stmt, nCol ), -1 );
      }
      else
      {
         value.setNumeric( odbc_column_double( m_stmt, nCol ) );
      }
      return true;

   case SQLITE_BLOB:
      {
         int len =  odbc_column_bytes( m_stmt, nCol );
         MemBuf* mb = new MemBuf_1( len );
         memcpy( mb->data(), (byte*) odbc_column_blob( m_stmt, nCol ), len );
         value = mb;
      }
      return true;


   case SQLITE_TEXT:
      {
         CoreString* cs = new CoreString;
         cs->fromUTF8( (const char*) odbc_column_text( m_stmt, nCol ) );
         value = cs;
      }
      return true;
   }

   return false;
}


/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementODBC::DBIStatementODBC( DBIHandleODBC *dbh, odbc_stmt* stmt ):
   DBIStatement( dbh ),
   m_statement( stmt ),
   m_inBind( stmt )
{
}

DBIStatementODBC::~DBIStatementODBC()
{
   close();
}

int64 DBIStatementODBC::execute( const ItemArray& params )
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   m_inBind.bind(params);
   int res = odbc_step( m_statement );
   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_EXEC, res );
   }

   // SQLite doesn't distinguish between fetch and insert statements; we do.
   // Exec never returns a recordset; instead, it is used to insert and re-issue
   // repeatedly statemnts. This is accomplished by Sqllite by issuing a reset
   // after each step.
   res = odbc_reset( m_statement );
   if( res != SQLITE_OK )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_EXEC, res );
   }

   return 0;
}

void DBIStatementODBC::reset()
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   int res = odbc_reset( m_statement );
   if( res != SQLITE_OK )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_RESET, res );
   }
}

void DBIStatementODBC::close()
{
   if( m_statement != 0 )
   {
      odbc_finalize( m_statement );
      m_statement = 0;
   }
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/

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

int64 DBIHandleODBC::getLastInsertedId()
{
   return -1;
}

int64 DBIHandleODBC::getLastInsertedId( const String& sequenceName )
{
   return -1;
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

void DBIHandleODBC::options( const String& params )
{
   if( m_settings.parse( params ) )
   {
      // To turn off the autocommit.
      SQLSetConnectAttr( m_conn->m_hHdbc, SQL_AUTOCOMMIT, 
            m_settings.m_bAutocommit ? SQL_AUTOCOMMIT_ON: SQL_AUTOCOMMIT_OFF, 
            0 );
   }
   else
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
   }
}

const DBISettingParams* DBIHandleODBC::options() const
{
   return &m_settings;
}

DBIRecordset *DBIHandleODBC::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   AutoCString asQuery( query );
   ODBCConn *conn = ((DBIHandleODBC *) m_dbh)->getConn();

   RETCODE ret = SQLExecDirect( conn->m_hHstmt, ( SQLCHAR* )asQuery.c_str( ), asQuery.length() );

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, conn->m_hHstmt, TRUE );
      // return
   }

   SQLINTEGER nRowCount;
   RETCODE retcode = SQLRowCount( conn->m_hHstmt, &nRowCount );

   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, conn->m_hHstmt, TRUE );
      // return
   }
   affectedRows = (int64) nRowCount;

   if( nRowCount != 0 )
   {
	   SQLSMALLINT nColCount;
	   retcode = SQLNumResultCols( conn->m_hHstmt, &nColCount );

      if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
	   {
  	      throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, conn->m_hHstmt, TRUE );
         // return
	   }

	   return new DBIRecordsetODBC( m_dbh, nRowCount, nColCount );
   }

   // query without recordset   
   throw new DBIError( ErrorParam(FALCON_DBI_ERROR_QUERY_EMPTY, __LINE__ ) );
   
   // to make the compiler happy
   return 0;
}

void DBIHandleODBC::perform( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   odbc_stmt* pStmt = int_prepare( sql );
   int_execute( pStmt, params );

   SQLINTEGER nRowCount;
   RETCODE retcode = SQLRowCount( conn->m_hHstmt, &nRowCount );

   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, conn->m_hHstmt, TRUE );
      // return
   }
   
   affectedRows = (int64) nRowCount;
}


DBIRecordset* DBIHandleODBC::call( const String &sql, int64 &affectedRows, const ItemArray& params )
{

   odbc_stmt* pStmt = int_prepare( sql );
   int count = odbc_column_count( pStmt );
   if( count == 0 )
   {
      int_execute( pStmt, params );
      affectedRows = odbc_changes( m_conn );
      return 0;
   }
   else
   {
      // the bindings must stay with the recordset...
      return new DBIRecordsetODBC( this, pStmt, params );
   }
}


DBIStatement* DBIHandleODBC::prepare( const String &query )
{
   odbc_stmt* pStmt = int_prepare( query );
   return new DBIStatementODBC( this, pStmt );
}


odbc_stmt* DBIHandleODBC::int_prepare( const String &sql ) const
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   AutoCString zSql( sql );
   odbc_stmt* pStmt = 0;
   int res = odbc_prepare_v2( m_conn, zSql.c_str(), zSql.length(), &pStmt, 0 );
   if( res != SQLITE_OK )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }

   return pStmt;
}

void DBIHandleODBC::int_execute( odbc_stmt* pStmt, const ItemArray& params )
{
   // int_execute is NEVER called alone
   fassert( m_conn != 0 );

   int res;
   if( params.length() > 0 )
   {
      Sqlite3InBind binds( pStmt );
      binds.bind(params);
      res = odbc_step( pStmt );
      odbc_finalize( pStmt );
   }
   else
   {
      res = odbc_step( pStmt );
      odbc_finalize( pStmt );
   }

   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }
}


void DBIHandleODBC::begin()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( !m_bInTrans )
   {
      SQLRETURN srRet = SQLEndTran( 
	      SQL_HANDLE_DBC, 
	      static_cast<DBIHandleODBC*>(m_dbh)->getConn()->m_hHdbc, 
	      SQL_COMMIT );

      m_inTransaction = false;

      if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
	      return dbi_error;
   }
}


void DBIHandleODBC::commit()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   ODBCConn* conn = static_cast<DBIHandleODBC*>(m_dbh)->getConn();

   SQLRETURN srRet = SQLEndTran( 
	   SQL_HANDLE_DBC, 
	   conn->m_hHdbc, 
	   SQL_COMMIT );

   m_inTransaction = false;

   if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
   {
      throwError( FALCON_DBI_ERROR_TRANSACTION, SQL_HANDLE_STMT, ->m_hHstmt, TRUE );
   }
}


void DBIHandleODBC::rollback()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
   
   ODBCConn* conn = static_cast<DBIHandleODBC*>(m_dbh)->getConn();

   SQLRETURN srRet = SQLEndTran( 
	   SQL_HANDLE_DBC, 
	   conn->m_hHdbc, 
	   SQL_ROLLBACK );

   m_inTransaction = false;
	
   if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
   {
      throwError( FALCON_DBI_ERROR_TRANSACTION, SQL_HANDLE_STMT, ->m_hHstmt, TRUE );
   }
}


void DBIHandleODBC::selectLimited( const String& query,
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
      result += " LIMIT " + sCount + sBegin;
   }
}




void DBIHandleODBC::close()
{
   if ( m_conn != NULL )
   {
      if( m_bInTrans )
      {
         odbc_exec( m_conn, "ROLLBACK", 0, 0, 0 );
         m_bInTrans = false;
      }

      odbc_close( m_conn );
      m_conn = NULL;
   }
}

//=====================================================================
// Utilities
//=====================================================================

void DBIHandleODBC::throwError( int falconError, SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd )
{
   String err = GetErrorMessage( plm_handle_type, GetErrorMessage, ConnInd );
   throw new DBIError( ErrorParam(falconError, __LINE__ ).extra(err) );
}

String DBIHandleODBC::GetErrorMessage(SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd)
{
	RETCODE     plm_retcode = SQL_SUCCESS;
	UCHAR       plm_szSqlState[MAXBUFLEN] = "";
   UCHAR       plm_szErrorMsg[MAXBUFLEN] = "";
	SDWORD      plm_pfNativeError = 0L;
	SWORD       plm_pcbErrorMsg = 0;
	SQLSMALLINT plm_cRecNmbr = 1;
	SDWORD      plm_SS_MsgState = 0, plm_SS_Severity = 0;
	SQLINTEGER  plm_Rownumber = 0;
	USHORT      plm_SS_Line;
	SQLSMALLINT plm_cbSS_Procname, plm_cbSS_Srvname;
	SQLCHAR     plm_SS_Procname[MAXNAME] ="", plm_SS_Srvname[MAXNAME] = "";
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


}

/* end of odbc_mod.cpp */

