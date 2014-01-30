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
#define SRC "modules/native/dbi/odbc/odbc_mod.cpp"

#include "odbc_mod.h"
#include <falcon/timestamp.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

#include <string.h>

#include <sqlucode.h>
#include <odbcss.h>

#include <stdio.h>

#define ODBC_COLNAME_SIZE 512

namespace Falcon
{

   
/******************************************************************************
 * Private class used to convert timestamp to MySQL format.
 *****************************************************************************/

class DBITimeConverter_ODBC_TIME: public DBITimeConverter
{
public:
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
} DBITimeConverter_ODBC_TIME_impl;

void DBITimeConverter_ODBC_TIME::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( ((unsigned)bufsize) >= sizeof( TIMESTAMP_STRUCT ) );

   TIMESTAMP_STRUCT* mtime = (TIMESTAMP_STRUCT*) buffer;
   mtime->year = (unsigned) ts->year();
   mtime->month = (unsigned) ts->month();
   mtime->day = (unsigned) ts->day();
   mtime->hour = (unsigned) ts->hour();
   mtime->minute = (unsigned) ts->minute();
   mtime->second = (unsigned) ts->second();
   mtime->fraction = (unsigned) ts->msec()*100000;

   bufsize = sizeof( TIMESTAMP_STRUCT );
}

/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

ODBCInBind::ODBCInBind( SQLHSTMT stmt, bool bigInt ):
      m_hStmt( stmt ),
      m_pLenInd( 0 ),
      m_pColInfo( 0 ),
      m_bUseBigInteger( bigInt )
{}

ODBCInBind::~ODBCInBind()
{
   // we don't own the handlers
   if ( m_pLenInd != 0 )
   {
      free( m_pLenInd );
   }

   /*if( m_pColInfo != 0 )
   {
      delete[] m_pColInfo;
   }
   */
}


void ODBCInBind::onFirstBinding( int size )
{
   m_pLenInd = (SQLLEN *) malloc( sizeof(SQLINTEGER) * size );
   memset( m_pLenInd, 0, sizeof(SQLLEN ) * size );
   //m_pColInfo = new ODBCColInfo[ size ];
}


void ODBCInBind::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];
   SQLLEN& pLenInd = m_pLenInd[num];

   // fill the call variables with consistent defaults
   SQLSMALLINT     InputOutputType = SQL_PARAM_INPUT;
   SQLSMALLINT     ValueType = SQL_C_CHAR;
   SQLSMALLINT     ParameterType = SQL_CHAR;
   SQLSMALLINT     ColSize = 0;
   SQLLEN          BufferLength = 0;
   SQLPOINTER      ParameterValuePtr;

   switch( item.type() )
   {
   // set to null
   case DBIBindItem::t_nil:
      pLenInd = SQL_NULL_DATA;
      ParameterValuePtr = 0;
      ColSize = 1;
      break;

   case DBIBindItem::t_bool:
      ValueType = SQL_C_BIT;
      ParameterType = SQL_BIT;
      ParameterValuePtr = (SQLPOINTER) item.asBoolPtr();
      ColSize = 1;
      break;

   case DBIBindItem::t_int:
      if( m_bUseBigInteger )
      {
         ValueType = SQL_C_SBIGINT;
         ParameterType = SQL_BIGINT;
         ParameterValuePtr = (SQLPOINTER) item.asIntegerPtr();
         ColSize = 19;
         pLenInd = sizeof(int64);
      }
      else
      {
         ValueType = SQL_C_LONG;
         ParameterType = SQL_INTEGER;
         ParameterValuePtr = (SQLPOINTER) item.asIntegerPtr();
         ColSize = 10;
         pLenInd = sizeof(long);
      }
      break;

   case DBIBindItem::t_double:
      ValueType = SQL_C_DOUBLE;
      ParameterType = SQL_DOUBLE;
      ParameterValuePtr = (SQLPOINTER) item.asDoublePtr();
      ColSize = 15;
      pLenInd = sizeof(double);
      break;

   case DBIBindItem::t_string:
      ValueType = SQL_C_WCHAR;
      ParameterType = SQL_WCHAR;
      // String::toWideString is granted to ensure space and put extra '\0' at the end.
      // Use the extra incoming '\0'
      pLenInd = BufferLength = item.asStringLen()*sizeof(wchar_t);
      ParameterValuePtr = (SQLPOINTER) item.asString();
      ColSize = (SQLSMALLINT) BufferLength;
      break;

   case DBIBindItem::t_buffer:
      ValueType = SQL_C_BINARY;
      ParameterType = SQL_BINARY;
      pLenInd = BufferLength = item.asStringLen();
      ParameterValuePtr = (SQLPOINTER) item.asBuffer();
      break;

   case DBIBindItem::t_time:
      ValueType = SQL_C_TIMESTAMP;
      ParameterType = SQL_TIMESTAMP;
      pLenInd = BufferLength = item.asStringLen();
      ParameterValuePtr = (SQLPOINTER) item.asBuffer();
      ColSize = (SQLSMALLINT) pLenInd;
      break;
   }

   SQLRETURN ret = SQLBindParameter( 
      m_hStmt,
      num+1,
      InputOutputType,
      ValueType,
      ParameterType,
      ColSize,
      0,
      ParameterValuePtr,
      BufferLength,
      &pLenInd);

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_BIND_INTERNAL, SQL_HANDLE_STMT, m_hStmt, TRUE, false );
   }
}


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetODBC::DBIRecordsetODBC( DBIHandleODBC *dbh, int64 nRowCount, int32 nColCount, ODBCStatementHandler* h )
    : DBIRecordset( dbh ),
      m_pStmt( h ),
      m_nRow( 0 ),
      m_nRowCount( nRowCount ),
      m_nColumnCount( nColCount ),
      m_pColInfo(0)
{
   dbh->incConnRef();
   m_conn = dbh->getConn();
   h->incref();
   m_bAsString = dbh->options()->m_bFetchStrings;
}


DBIRecordsetODBC::DBIRecordsetODBC( DBIHandleODBC *dbh, int64 nRowCount, int32 nColCount, SQLHSTMT hStmt )
    : DBIRecordset( dbh ),
      m_pStmt( new ODBCStatementHandler( hStmt) ),
      m_nRow( 0 ),
      m_nRowCount( nRowCount ),
      m_nColumnCount( nColCount ),
      m_pColInfo(0)
{
   dbh->incConnRef();
   m_conn = dbh->getConn();
   m_bAsString = dbh->options()->m_bFetchStrings;
}


DBIRecordsetODBC::~DBIRecordsetODBC()
{   
   close();
   delete[] m_pColInfo;
}

int DBIRecordsetODBC::getColumnCount()
{
   return m_nColumnCount;
}

int64 DBIRecordsetODBC::getRowIndex()
{
   return m_nRow;
}

int64 DBIRecordsetODBC::getRowCount()
{
   return m_nRowCount;
}


bool DBIRecordsetODBC::getColumnName( int nCol, String& name )
{
   if( m_pStmt == 0 )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__, SRC ) );
   }

   // a good moment to fetch the column data
   if ( m_pColInfo == 0 )
   {
      GetColumnInfo();
   }

   if ( nCol < 0 || nCol >= m_nColumnCount )
   {
      return false;
   }
   
   name = m_pColInfo[nCol].sName;
   return true;
}


void DBIRecordsetODBC::GetColumnInfo()
{
   m_pColInfo = new ODBCColInfo[ m_nColumnCount ];
   
   wchar_t ColumnName[ODBC_COLNAME_SIZE];
   SQLSMALLINT    NameLength;
   SQLHSTMT hStmt = m_pStmt->handle();

   for ( SQLSMALLINT nCol = 0; nCol < m_nColumnCount; ++nCol )
   {
      ODBCColInfo& current = m_pColInfo[nCol];

      SQLRETURN ret = SQLDescribeColW( 
         hStmt, 
         nCol+1,  
         ColumnName,
         ODBC_COLNAME_SIZE,
         &NameLength,
         &current.DataType,
         &current.ColumnSize,
         &current.DecimalDigits,
         &current.Nullable
       );

      if ( NameLength+1 > ODBC_COLNAME_SIZE )
      {
         NameLength = ODBC_COLNAME_SIZE - 1;
      }

      if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
      {
         // Vital informations are not available; better bail out and close
         DBIError* dbie = new DBIError( ErrorParam( FALCON_DBI_ERROR_FETCH, __LINE__, SRC ).
            extra( DBIHandleODBC::GetErrorMessage( SQL_HANDLE_STMT, hStmt, TRUE ) ) );
         close();

         throw dbie;
         // return
      }

      ColumnName[NameLength] = 0;
      current.sName = ColumnName;
      current.sName.bufferize();
   }
}


bool DBIRecordsetODBC::fetchRow()
{
   if( m_pStmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__, SRC ) );

   SQLRETURN ret = SQLFetch( m_pStmt->handle() );
   if ( ret == SQL_NO_DATA )
   {
      // we're done
      return false;
   }

   // error?
   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      // throw but don't close
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_FETCH, SQL_HANDLE_STMT, m_pStmt->handle(), TRUE, false );
      // return
   }

   // more data incoming
   m_nRow++;
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
   if( m_pStmt != 0 )
   {
      m_pStmt->decref();
      m_conn->decref();
      m_pStmt = 0;
   }
}


bool DBIRecordsetODBC::getColumnValue( int nCol, Item& value )
{
   static Class* clsTS = Engine::instance()->stdHandlers()->timestampClass();

   if( m_pStmt == 0 )
   {
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__, SRC ) );
   }

   if ( nCol < 0 || nCol >= m_nColumnCount )
   {
      return false;
   }
   
   // a good moment to fetch the column data
   if ( m_pColInfo == 0 )
   {
      GetColumnInfo();
   }

   SQLHSTMT hStmt = m_pStmt->handle();
   ODBCColInfo& column = m_pColInfo[ nCol ];
   
   SQLRETURN ret;
   int64 integer;
   double real;
   unsigned char uchar;
   SQLLEN ExpSize = 0, nSize = 0;

   // do a first call to determine null or memory requirements
   switch ( column.DataType )
   {
   case SQL_CHAR:
   case SQL_VARCHAR:
   case SQL_LONGVARCHAR:
   case SQL_WCHAR:
   case SQL_WVARCHAR:
   case SQL_WLONGVARCHAR:
      ret = SQLGetData( hStmt, nCol+1, SQL_C_WCHAR, &uchar, 0, &ExpSize);
      if( ret != SQL_ERROR )
      {
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else if( ExpSize == 0 )
         {
            value = FALCON_GC_HANDLE(new String(""));
         }
         else
         {
            // we must account for an extra '0' put in by ODBC
            uint32 alloc = ExpSize+sizeof(wchar_t);
            wchar_t *cStr = (wchar_t*) malloc( alloc );  
            ret = SQLGetData( hStmt, nCol+1, SQL_C_WCHAR, cStr, alloc, &nSize);

            // save the data even in case we had an error, or we'll leak
            String* cs = new String;
            uint32 size = ExpSize/sizeof(wchar_t);
            cs->adopt( cStr, size, alloc );
            value = FALCON_GC_HANDLE(cs);
         }
      }
      break;

   case SQL_DECIMAL:
   case SQL_NUMERIC:
   case SQL_REAL:
   case SQL_FLOAT:
   case SQL_DOUBLE:
      if( m_bAsString )
      {
         char buffer[32];
         ret = SQLGetData( hStmt, nCol+1, SQL_C_CHAR, &buffer, sizeof(buffer), &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else {
            String* s = new String(buffer);
            s->bufferize();
            value = FALCON_GC_HANDLE(s);
         }
      }
      else 
      {
         ret = SQLGetData( hStmt, nCol+1, SQL_C_DOUBLE, &real, sizeof(real), &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else {
            value.setNumeric( real );
         }
      }
      break;

   case SQL_INTERVAL_MONTH:	
   case SQL_INTERVAL_YEAR:
   case SQL_INTERVAL_DAY:
   case SQL_INTERVAL_YEAR_TO_MONTH:
   case SQL_INTERVAL_HOUR:
   case SQL_INTERVAL_MINUTE:
   case SQL_INTERVAL_SECOND:
   case SQL_INTERVAL_DAY_TO_HOUR:
   case SQL_INTERVAL_DAY_TO_MINUTE:
   case SQL_INTERVAL_DAY_TO_SECOND:
   case SQL_INTERVAL_HOUR_TO_MINUTE:
   case SQL_INTERVAL_HOUR_TO_SECOND:
   case SQL_INTERVAL_MINUTE_TO_SECOND:   
   case SQL_GUID:
   case SQL_TINYINT:
   case SQL_SMALLINT:
   case SQL_INTEGER:
   case SQL_BIGINT:
      if( m_bAsString )
      {
         char buffer[32];
         ret = SQLGetData( hStmt, nCol+1, SQL_C_CHAR, &buffer, sizeof(buffer), &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else {
            String* s = new String(buffer);
            s->bufferize();
            value = FALCON_GC_HANDLE(s);
         }
      }
      else
      {
         ret = SQLGetData( hStmt, nCol+1, SQL_C_SBIGINT, &integer, sizeof(integer), &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else {
            value.setInteger( integer );
         }
      }
      break;
   
   case SQL_BIT:
      ret = SQLGetData( hStmt, nCol+1, SQL_C_BIT, &uchar, sizeof(uchar), &ExpSize);
      if( ExpSize == SQL_NULL_DATA )
      {
         value.setNil();
      }
      else
      {
         if( m_bAsString )
         {
            value = FALCON_GC_HANDLE( new String( uchar ? "true" : "false" ) );
         }
         else
         {
            value.setBoolean( uchar ? true : false );
         }
      }
      return true;

   case SQL_BINARY:
   case SQL_VARBINARY:
   case SQL_LONGVARBINARY:
      ret = SQLGetData( hStmt, nCol+1, SQL_C_BINARY, &uchar, 0, &ExpSize);
      if( ret != SQL_ERROR )
      {
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
            return true;
         }

         String* s = new String(ExpSize);
         s->reserve(ExpSize);
         ret = SQLGetData( hStmt, nCol+1, SQL_C_BINARY, s->getRawStorage(), ExpSize , &nSize);
         s->size(nSize);
         s->toMemBuf();
         // save the data nevertheless
         value = FALCON_GC_HANDLE(s);
      }
      break;

   case SQL_TYPE_DATE:
   case SQL_TYPE_TIME:	
   case SQL_TYPE_TIMESTAMP:
      if( m_bAsString )
      {
         char buffer[32];
         ret = SQLGetData( hStmt, nCol+1, SQL_C_CHAR, &buffer, sizeof(buffer), &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else {
            String* s = new String(buffer);
            s->bufferize();
            value = FALCON_GC_HANDLE(s);
         }
      }
      else
      {
         TIMESTAMP_STRUCT tstamp;
         ret = SQLGetData( hStmt, nCol+1, SQL_C_TYPE_TIMESTAMP, &tstamp, sizeof(tstamp) , &ExpSize);
         if( ExpSize == SQL_NULL_DATA )
         {
            value.setNil();
         }
         else if ( ret == SQL_SUCCESS || ret == SQL_SUCCESS_WITH_INFO )
         {
            TimeStamp* ts = new TimeStamp;

            ts->set( tstamp.year,
               tstamp.month,
               tstamp.day,
               tstamp.hour,            
               tstamp.minute,
               tstamp.second,
               (int16) tstamp.fraction/100000 );

            value = FALCON_GC_STORE( clsTS, ts );
         }
      }
      break;

   default:
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_UNHANDLED_TYPE, __LINE__, SRC ) );
   }

   if ( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_FETCH, SQL_HANDLE_STMT, hStmt, TRUE, false );
   }

   return true;
}


/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementODBC::DBIStatementODBC( DBIHandleODBC *dbh, SQLHSTMT hStmt ):
   DBIStatement( dbh ),
   m_inBind( hStmt, dbh->options()->m_bUseBigInt ),
   m_pStmt( new ODBCStatementHandler(hStmt) )
{
   dbh->incConnRef();
   m_conn = dbh->getConn();
}

DBIStatementODBC::~DBIStatementODBC()
{
   close();
}


DBIRecordset* DBIStatementODBC::execute( ItemArray* params )
{
   if( m_pStmt == 0 )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__, SRC ) );
   }

   if( params != 0 )
   {
      m_inBind.bind(*params, DBITimeConverter_ODBC_TIME_impl, DBIStringConverter_WCHAR_impl );
   }


   SQLHSTMT hStmt = m_pStmt->handle();
   SQLRETURN ret = SQLExecute( hStmt );

   if ( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_FETCH, SQL_HANDLE_STMT, hStmt, TRUE, false );
   }

   // Cont the rows
   SQLLEN nRowCount;
   RETCODE retcode = SQLRowCount( hStmt, &nRowCount );
   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {      
  	   DBIHandleODBC::throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE, false );
      // return
   }

   m_nLastAffected = (int64) nRowCount;

   // create the recordset
   SQLSMALLINT ColCount;
   retcode = SQLNumResultCols( hStmt, &ColCount);
   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {      
  	   DBIHandleODBC::throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE, false );
      // return
   }

   // Do we have a recordset?
   if ( ColCount > 0 )
   {
      // this may throw -- and in that case will close hStmt
      return new DBIRecordsetODBC( static_cast<DBIHandleODBC*>(m_dbh), nRowCount, ColCount, m_pStmt );
   }
   else 
   {
      return 0;
   }	
}

void DBIStatementODBC::reset()
{
   if( m_pStmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__, SRC ) );
}

void DBIStatementODBC::close()
{
   if( m_pStmt != 0 )
   {
      m_pStmt->decref();
      m_conn->decref();
      m_pStmt = 0;
   }
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleODBC::DBIHandleODBC( const Class* h ):
   DBIHandle(h)
{
	m_conn = NULL;
   m_bInTrans = false;
}

DBIHandleODBC::~DBIHandleODBC( )
{
	close( );
}


ODBCConn *DBIHandleODBC::getConnData()
{
   if( m_conn == 0 )
   {
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__, SRC ) );
   }

   return m_conn;
}


int64 DBIHandleODBC::getLastInsertedId( const String& sequenceName )
{
   ODBCConn *conn = getConnData();
   
   // It's a trick, but it should work
   SQLHSTMT hStmt = openStatement( conn->m_hHdbc );
   SQLRETURN ret = SQLExecDirectA( hStmt, (SQLCHAR*) "SELECT @@IDENTITY", SQL_NTS );

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
   }
   
   // Cont the rows
   ret = SQLFetch( hStmt );
   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
   }

   int64 value;
   SQLLEN ind;
   ret = SQLGetData( hStmt, 1, SQL_C_SBIGINT, &value, sizeof(value), &ind );
   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
   }

   SQLFreeStmt( hStmt, SQL_CLOSE );
   return value;
}


void DBIHandleODBC::close()
{
	if( m_conn )
	{
		m_conn->decref();
		m_conn = 0;
	}
}


void DBIHandleODBC::connect( const String &parameters )
{
   AutoCString asConnParams( parameters );

   SQLHENV hEnv;
   SQLHDBC hHdbc;

   RETCODE retcode = SQLAllocHandle (SQL_HANDLE_ENV, NULL, &hEnv);

   if( ( retcode != SQL_SUCCESS_WITH_INFO ) && ( retcode != SQL_SUCCESS ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__, SRC )
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
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__, SRC )
         .extra( "Impossible to allocate ODBC connection handle and connect." ));
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

      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__, SRC)
         .extra( errorMessage ));
   }

   ODBCConn* conn = new ODBCConn( hEnv, hHdbc );
   m_conn = conn;
}


void DBIHandleODBC::options( const String& params )
{
   ODBCConn* conn = getConnData();

   if( m_settings.parse( params ) )
   {
      // To turn off the autocommit.
      SQLINTEGER commitValue = m_settings.m_bAutocommit ? SQL_AUTOCOMMIT_ON: SQL_AUTOCOMMIT_OFF;
      SQLSetConnectAttr( conn->m_hHdbc, SQL_AUTOCOMMIT, &commitValue, SQL_IS_INTEGER );
   }
   else
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__, SRC )
            .extra( params ) );
   }
}

const DBISettingParamsODBC* DBIHandleODBC::options() const
{
   return &m_settings;
}

   
SQLHSTMT DBIHandleODBC::openStatement(SQLHDBC hHdbc) 
{
   SQLHSTMT hHstmt;
   SQLRETURN retcode = SQLAllocHandle( SQL_HANDLE_STMT, hHdbc, &hHstmt );

   if( ( retcode != SQL_SUCCESS ) && ( retcode != SQL_SUCCESS_WITH_INFO ) )
   {
      // don't close the db for this
      throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_DBC, hHdbc, TRUE, false );
   }

   return hHstmt;
}

/*
SQLHDESC DBIHandleODBC::getStatementDesc( SQLHSTMT hHstmt )
{
   ODBCConn *conn = ((DBIHandleODBC *) m_dbh)->getConn();
   SQLHDESC hIpd;

   retcode = SQLGetStmtAttr( hHstmt, SQL_ATTR_IMP_PARAM_DESC, &hIpd, 0, 0 );

   if( ( retcode != SQL_SUCCESS ) && ( retcode != SQL_SUCCESS_WITH_INFO ) )
   {
      // will close hHstmt
      throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_DBC, hHstmt, TRUE, true );
   }

   return hIpd;
}
*/

DBIRecordset *DBIHandleODBC::query( const String &sql, ItemArray* params )
{
   ODBCConn *conn = getConnData();
   SQLHSTMT hStmt = openStatement( conn->m_hHdbc );
   SQLRETURN ret;
   
   AutoWString asQuery( sql );

   // call the query
   if( params == 0 )
   {
      // -- no params -- easier.
      ret = SQLExecDirectW( hStmt, ( SQLWCHAR* )asQuery.w_str(), asQuery.length() );
   }
   else 
   {
      ret = SQLPrepareW( hStmt, (SQLWCHAR*) asQuery.w_str(), asQuery.length() );
      if ( ret != SQL_ERROR )
      {
         ODBCInBind inBind( hStmt, options()->m_bUseBigInt );
         inBind.bind(*params, DBITimeConverter_ODBC_TIME_impl, DBIStringConverter_WCHAR_impl );
         ret = SQLExecute( hStmt );
      }
   }

   if( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
      // return
   }
   
   // Cont the rows
   SQLLEN nRowCount;
   RETCODE retcode = SQLRowCount( hStmt, &nRowCount );

   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {      
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
      // return
   }

   m_nLastAffected = (int64) nRowCount;

   // create the recordset
   SQLSMALLINT ColCount;
   retcode = SQLNumResultCols( hStmt, &ColCount);
   if( retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO )
   {      
  	   throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
      // return
   }

   // Do we have a recordset?
   if ( ColCount > 0 )
   {
      // this may throw -- and in that case will close hStmt
      return new DBIRecordsetODBC( this, nRowCount, ColCount, hStmt );
   }
   else 
   {
      return 0;
   }	
}



DBIStatement* DBIHandleODBC::prepare( const String &query )
{
   ODBCConn *conn = getConnData();
   SQLHSTMT hStmt = openStatement( conn->m_hHdbc );
   
   AutoWString wQuery( query );
   SQLRETURN ret = SQLPrepareW( hStmt, (SQLWCHAR*) wQuery.w_str(), wQuery.length() );
   if ( ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO )
   {
      throwError( FALCON_DBI_ERROR_QUERY, SQL_HANDLE_STMT, hStmt, TRUE );
   }

   return new DBIStatementODBC( this, hStmt );
}


void DBIHandleODBC::begin()
{
   ODBCConn *conn = getConnData();

   if( m_bInTrans )
   {
      SQLRETURN srRet = SQLEndTran( SQL_HANDLE_DBC, conn->m_hHdbc, SQL_COMMIT );
      if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
      {
         m_bInTrans = false;
         throwError( FALCON_DBI_ERROR_TRANSACTION, SQL_HANDLE_DBC, conn->m_hHdbc, TRUE, false );
      }
   }
   
   m_bInTrans = true;
}


void DBIHandleODBC::commit()
{
   ODBCConn *conn = getConnData();

   if( m_bInTrans )
   {
      SQLRETURN srRet = SQLEndTran( SQL_HANDLE_DBC, conn->m_hHdbc, SQL_COMMIT );
      if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
      {
         m_bInTrans = false;
         throwError( FALCON_DBI_ERROR_TRANSACTION, SQL_HANDLE_DBC, conn->m_hHdbc, TRUE, false );
      }
   }

    m_bInTrans = false;
}


void DBIHandleODBC::rollback()
{
   ODBCConn *conn = getConnData();

   if( m_bInTrans )
   {
      SQLRETURN srRet = SQLEndTran( SQL_HANDLE_DBC, conn->m_hHdbc, SQL_ROLLBACK );
      m_bInTrans = false;
      if ( srRet != SQL_SUCCESS && srRet != SQL_SUCCESS_WITH_INFO )
      {
         throwError( FALCON_DBI_ERROR_TRANSACTION, SQL_HANDLE_DBC, conn->m_hHdbc, TRUE, false );
      }
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

//============================================================
// Settings parameter parser
//============================================================
DBISettingParamsODBC::DBISettingParamsODBC():
      m_bUseBigInt( false )
{
   addParameter( "bigint", m_sUseBigint );
}

DBISettingParamsODBC::DBISettingParamsODBC( const DBISettingParamsODBC & other):
   DBISettingParams(other),
   m_bUseBigInt( other.m_bUseBigInt )
{
   // we don't care about the parameter parsing during the copy.
}


DBISettingParamsODBC::~DBISettingParamsODBC()
{
}


bool DBISettingParamsODBC::parse( const String& connStr )
{
   if( ! DBISettingParams::parse(connStr) )
   {
      return false;
   }

   return checkBoolean( m_sUseBigint, m_bUseBigInt );
}


//=====================================================================
// Utilities
//=====================================================================

void DBIHandleODBC::throwError( int falconError, SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd, bool free )
{
   String err = GetErrorMessage( plm_handle_type, plm_handle, ConnInd );
   if (free)
   {
      SQLFreeHandle( plm_handle_type, plm_handle );
   }
   throw new DBIError( ErrorParam(falconError, __LINE__, SRC ).extra(err) );
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
	SQLCHAR     plm_SS_Procname[ODBC_COLNAME_SIZE] ="", plm_SS_Srvname[ODBC_COLNAME_SIZE] = "";
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

