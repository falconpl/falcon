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
#include <sql.h>

namespace Falcon
{
const unsigned long MAXBUFLEN = 256;
class DBIHandleODBC;

class ODBCConn
{
public:
	SQLHENV m_hEnv;
	SQLHDBC m_hHdbc;

   ODBCConn()
   {
      m_hEnv = SQL_NULL_HENV;
		m_hHdbc = SQL_NULL_HDBC;
      m_refCount = 1;
   }

	ODBCConn( SQLHENV hEnv, SQLHDBC hHdbc )
	{
		m_hEnv = hEnv;
		m_hHdbc = hHdbc;
      m_refCount = 1;
	}

   void incref() { m_refCount++; }
   void decref() { if ( --m_refCount == 0 ) delete this; }

private:
   ~ODBCConn()
	{
		if( m_hHdbc != SQL_NULL_HDBC )
		{
			SQLDisconnect( m_hHdbc );
			SQLFreeHandle( SQL_HANDLE_DBC, m_hHdbc );
		}

		if( m_hEnv != SQL_NULL_HENV)
			SQLFreeHandle(SQL_HANDLE_ENV, m_hEnv );
	}

   int m_refCount;
};



class ODBCInBind: public DBIInBind
{
public:
   ODBCInBind(SQLHSTMT hStmt, bool bBigInt );
   virtual ~ODBCInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

protected:
   class ODBCColInfo 
   {
   public:
      SQLSMALLINT    DataType;
      SQLULEN        ColumnSize;
      SQLSMALLINT    DecimalDigits;
      SQLSMALLINT    Nullable;
   };

   SQLHSTMT m_hStmt;
   SQLLEN* m_pLenInd;
   ODBCColInfo *m_pColInfo;
   bool m_bUseBigInteger;
};


class ODBCStatementHandler: public DBIRefCounter<SQLHSTMT> {
public:
   
   ODBCStatementHandler( SQLHSTMT hStmt ):
      DBIRefCounter( hStmt )
   {}

   ~ODBCStatementHandler() {
      SQLFreeStmt( handle(), SQL_CLOSE );
   }
};


class DBIRecordsetODBC : public DBIRecordset
{
public:
   DBIRecordsetODBC( DBIHandleODBC *dbt, int64 nRowCount, int32 nColCount, ODBCStatementHandler* h );
   DBIRecordsetODBC( DBIHandleODBC *dbt, int64 nRowCount, int32 nColCount, SQLHSTMT h );
   virtual ~DBIRecordsetODBC();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();

   SQLHSTMT handle() const { return m_pStmt->handle(); }
   
protected:
   class ODBCColInfo 
   {
   public:
      String sName;
      SQLSMALLINT    DataType;
      SQLULEN        ColumnSize;
      SQLSMALLINT    DecimalDigits;
      SQLSMALLINT    Nullable;
   };

   ODBCStatementHandler* m_pStmt;   
   int64 m_nRow;
   int64 m_nRowCount;
   int32 m_nColumnCount;
   
   ODBCColInfo* m_pColInfo;
   bool m_bAsString;
   ODBCConn *m_conn;

   void GetColumnInfo();

};


class DBIStatementODBC: public DBIStatement
{
public:
   DBIStatementODBC( DBIHandleODBC *dbh, SQLHSTMT h );
   virtual ~DBIStatementODBC();

   virtual DBIRecordset* execute( ItemArray* params = 0 );
   virtual void reset();
   virtual void close();

   SQLHSTMT handle() const { return m_pStmt->handle(); }
protected:
   ODBCInBind m_inBind;
   ODBCStatementHandler* m_pStmt;
   ODBCConn *m_conn;
};


class DBISettingParamsODBC: public DBISettingParams
{
private:
   String m_sUseBigint;

public:
   DBISettingParamsODBC();
   DBISettingParamsODBC( const DBISettingParamsODBC & other );
   virtual ~DBISettingParamsODBC();

   /** Specific parse analizying the options */
   virtual bool parse( const String& connStr );

   /** True if we can use int64 on the underlying driver. */
   bool m_bUseBigInt;
};

class DBIHandleODBC : public DBIHandle
{
protected:
	ODBCConn* m_conn;
   DBISettingParamsODBC m_settings;
   bool m_bInTrans;
   /** Checks if the connection is open and throws otherwise */
   ODBCConn *getConnData();

   SQLHSTMT openStatement(SQLHDBC hdbc);
   SQLHDESC getStatementDesc( SQLHSTMT hHstmt );
   
   

public:
   DBIHandleODBC( const Class* h );
   virtual ~DBIHandleODBC();

   virtual void connect( const String& params );
   virtual void options( const String& params );
   virtual const DBISettingParamsODBC* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, ItemArray* params );
   virtual DBIStatement* prepare( const String &query );
   virtual int64 getLastInsertedId( const String& name = "" );

   virtual void begin();
   virtual void commit();
   virtual void rollback();

   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result );

   ODBCConn *getConn() { return m_conn; }

   /** Throws a DBI error wsrapping an ODBC error. */
   static void throwError( int falconError, SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd, bool free = true );
   
   /** Utility to get ODBC error description. */
   static String GetErrorMessage(SQLSMALLINT plm_handle_type, SQLHANDLE plm_handle, int ConnInd);

   void incConnRef() { m_conn->incref(); }
   void decConnRef() { m_conn->decref(); }
};

}

#endif /* DBI_ODBC_H */

/* end of odbc.h */

