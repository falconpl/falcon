/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_mod.h
 *
 * ODBC driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Tue Sep 30 17:00:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_DBI_ODBC_H
#define FALCON_DBI_ODBC_H

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>
#include <sql.h>

namespace Falcon
{
const unsigned long MAXBUFLEN = 256;

String GetErrorMessage(SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd);

struct ODBCConn
{
	SQLHENV m_hEnv;
	SQLHDBC m_hHdbc;
	SQLHSTMT m_hHstmt;
	SQLHDESC m_hIpd;

	void Initialize( )
	{
		m_hEnv = SQL_NULL_HENV;
		m_hHdbc = SQL_NULL_HDBC;
		m_hHstmt = SQL_NULL_HSTMT;
		m_hIpd = SQL_NULL_HDESC;
	}

	void Initialize( const SQLHENV hEnv, const SQLHDBC hHdbc, const SQLHSTMT hHstmt, const SQLHDESC hIpd )
	{
		m_hEnv = hEnv;
		m_hHdbc = hHdbc;
		m_hHstmt = hHstmt;
		m_hIpd = hIpd;
	}

	void Destroy( )
	{
		if( m_hIpd != SQL_NULL_HDESC )
			SQLFreeHandle(SQL_HANDLE_DESC, m_hIpd );

		if( m_hHstmt != SQL_NULL_HSTMT )
			SQLFreeHandle( SQL_HANDLE_STMT, m_hHstmt );

		if( m_hHdbc != SQL_NULL_HDBC )
		{
			SQLDisconnect( m_hHdbc );
			SQLFreeHandle( SQL_HANDLE_DBC, m_hHdbc );
		}

		if( m_hEnv != SQL_NULL_HENV)
			SQLFreeHandle(SQL_HANDLE_ENV, m_hEnv );
	}
};


class DBIRecordsetODBC : public DBIRecordset
{
public:
	struct SRowData 
	{
		void* m_pData;
		int m_nLen;
	};

	SRowData* m_pDataArr;

protected:
   ODBCConn* m_pConn;
   int m_nRow;
   int m_nRowCount;
   int m_nColumnCount;
   String m_sLastError;

protected:
   int m_row;
   int m_columnCount;
   sqlite3_stmt *m_stmt;
   Sqlite3InBind m_bind;
   bool m_bAsString;

public:
   DBIRecordsetSQLite3( DBIHandleSQLite3 *dbt, sqlite3_stmt* stmt, const ItemArray& inBind );
   virtual ~DBIRecordsetSQLite3();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();
};


class DBIStatementSQLite3: public DBIStatement
{
protected:
   sqlite3_stmt* m_statement;
   Sqlite3InBind m_inBind;

public:
   DBIStatementSQLite3( DBIHandleSQLite3 *dbh, sqlite3_stmt* stmt );
   virtual ~DBIStatementSQLite3();

   virtual int64 execute( const ItemArray& params );
   virtual void reset();
   virtual void close();

   sqlite3_stmt* sqlite3_statement() const { return m_statement; }
};


class DBIStatementODBC: public DBIStatement
{
protected:
   void* m_statement;

public:
   DBIStatementODBC( DBIHandleODBC *dbh, void* stmt );
   virtual ~DBIStatementODBC();

   virtual int64 execute( const ItemArray& params );
   virtual void reset();
   virtual void close();

   void* odbc_statement() const { return m_statement; }
};


class DBIHandleODBC : public DBIHandle
{
protected:
	ODBCConn* m_conn;
   DBISettingParams m_settings;
   bool m_bInTrans;

public:
   DBIHandleODBC();
   DBIHandleODBC( ODBCConn *conn );
   virtual ~DBIHandleODBC();

   virtual void options( const String& params );
   virtual const DBISettingParams* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual void perform( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual DBIRecordset* call( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual DBIStatement* prepare( const String &query );
   virtual int64 getLastInsertedId( const String& name = "" );

   virtual void begin();
   virtual void commit();
   virtual void rollback();

   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result );

   static void throwError( int falconError, int sql3Error, char* edesc=0 );
   static String errorDesc( int error );
   sqlite3 *getConn() { return m_conn; }
};


class DBIServiceODBC : public DBIService
{
public:
   DBIServiceODBC() : DBIService( "DBI_odbc" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceODBC theODBCService;

#endif /* DBI_ODBC_H */

/* end of odbc.h */

