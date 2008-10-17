/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_mod.h
 *
 * MySQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Tiziano De Rubeis
 * Begin: Tue Sep 30 17:00:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_ODBC_H
#define DBI_ODBC_H

#include "../include/dbiservice.h"
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

	   static dbi_type getFalconType( int typ );

	public:
	   DBIRecordsetODBC( DBIHandle *dbh, int nRows, int nCols );
	   ~DBIRecordsetODBC();

	   virtual dbi_status next();
	   virtual int getRowCount();
	   virtual int getRowIndex();
	   virtual int getColumnCount();
	   virtual dbi_status getColumnNames( char *names[] );
	   virtual dbi_status getColumnTypes( dbi_type *types );
	   virtual dbi_status asString( const int columnIndex, String &value );
	   virtual dbi_status asBoolean( const int columnIndex, bool &value );
	   virtual dbi_status asInteger( const int columnIndex, int32 &value );
	   virtual dbi_status asInteger64( const int columnIndex, int64 &value );
	   virtual dbi_status asNumeric( const int columnIndex, numeric &value );
	   virtual dbi_status asDate( const int columnIndex, TimeStamp &value );
	   virtual dbi_status asTime( const int columnIndex, TimeStamp &value );
	   virtual dbi_status asDateTime( const int columnIndex, TimeStamp &value );
	   virtual dbi_status asBlobID( const int columnIndex, String &value );
	   virtual void close();
	   virtual dbi_status getLastError( String &description );
	   virtual dbi_status DBIRecordsetODBC::bind( int ord, int type );
	};


	class DBITransactionODBC : public DBITransaction
	{
	protected:
	   bool m_inTransaction;
	   String m_sLastError;

	public:
	   DBITransactionODBC( DBIHandle *dbh );

	   virtual DBIRecordset *query( const String &query, dbi_status &retval );
	   virtual int execute( const String &query, dbi_status &retval );
	   virtual dbi_status begin();
	   virtual dbi_status commit();
	   virtual dbi_status rollback();
	   virtual void close();
	   virtual dbi_status getLastError( String &description );

	   virtual DBIBlobStream *openBlob( const String &blobId, dbi_status &status );
	   virtual DBIBlobStream *createBlob( dbi_status &status, const String &params= "",
		  bool bBinary = false );
	};

	class DBIHandleODBC : public DBIHandle
	{
	protected:
		ODBCConn* m_conn;
		DBITransactionODBC *m_connTr;

	public:
		DBIHandleODBC();
		DBIHandleODBC( ODBCConn *conn );
		virtual ~DBIHandleODBC();

		ODBCConn *getConn() { return m_conn; }

		virtual DBITransaction *startTransaction();
		virtual dbi_status closeTransaction( DBITransaction *tr );
		virtual DBIRecordset *query( const String &sql, dbi_status &retval );
		virtual int execute( const String &sql, dbi_status &retval );
		virtual int64 getLastInsertedId();
		virtual int64 getLastInsertedId( const String &value );
		virtual dbi_status getLastError( String &description );
		virtual dbi_status escapeString( const String &value, String &escaped );
		virtual dbi_status close();
	};

	class DBIServiceODBC : public DBIService
	{
	public:
	   DBIServiceODBC() : DBIService( "DBI_odbc" ) {}

	   virtual dbi_status init();
	   virtual DBIHandle *connect( const String &parameters, bool persistent,
								   dbi_status &retval, String &errorMessage );
	   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
	};

}

extern Falcon::DBIServiceODBC theODBCService;

#endif /* DBI_ODBC_H */

/* end of odbc.h */

