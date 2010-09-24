/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_mod.h
 *
 * Firebird driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 16:58:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_FBSQL_H
#define FALCON_FBSQL_H

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

#include <ibase.h>
#include <ib_util.h>
#include <iberror.h>

namespace Falcon
{

class FirebirdDBIInBind: public DBIInBind
{

public:
   FirebirdDBIInBind( isc_stmt_handle* stmt );
   virtual ~FirebirdDBIInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   XSQLDA* fb_bindings() const { return m_fbbind; }

private:
   XSQLDA* m_fbbind;
   isc_stmt_handle* m_stmt;
};


class FirebirdDBIOutBind: public DBIOutBind
{
public:
   FirebirdDBIOutBind():
      bIsNull( false ),
      nLength( 0 )
   {}

   ~FirebirdDBIOutBind() {}

   my_bool bIsNull;
   unsigned long nLength;
};

class DBIHandleFirebird;

class DBIRecordsetFirebird: public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   MYSQL_RES *m_res;
   MYSQL_FIELD* m_fields;

   bool m_bCanSeek;
public:
   DBIRecordsetFirebird( DBIHandleFirebird *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetFirebird();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual void close();
};

class DBIRecordsetFirebird_STMT: public DBIRecordsetFirebird
{
protected:
   MYSQL_STMT *m_stmt;

   // Binding data
   MYSQL_BIND* m_pMyBind;
   FirebirdDBIOutBind* m_pOutBind;

   // used to keep track of blobs that must be zeroed before fetch
   int* m_pBlobId;
   int m_nBlobCount;

public:
   DBIRecordsetFirebird_STMT( DBIHandleFirebird *dbt, MYSQL_RES *res, MYSQL_STMT *stmt, bool bCanSeek = false );
   virtual ~DBIRecordsetFirebird_STMT();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();
};


class DBIRecordsetFirebird_RES : public DBIRecordsetFirebird
{
protected:
   MYSQL_ROW m_rowData;
   CoreObject* makeTimestamp( const String& str );

public:
   DBIRecordsetFirebird_RES( DBIHandleFirebird *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetFirebird_RES();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
};

class DBIRecordsetFirebird_RES_STR: public DBIRecordsetFirebird_RES
{
public:
   DBIRecordsetFirebird_RES_STR( DBIHandleFirebird *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetFirebird_RES_STR();

   virtual bool getColumnValue( int nCol, Item& value );
};


class DBIHandleFirebird : public DBIHandle
{
protected:
   isc_db_handle m_conn;
   isc_tr_handle m_tr;

   DBISettingParams m_settings;

   isc_stmt_handle* fb_prepare( const String &query );
   int64 fb_execute( isc_stmt_handle* stmt, FirebirdDBIInBind& bindings, const ItemArray& params );

public:
   DBIHandleFirebird();
   DBIHandleFirebird( const isc_db_handle &conn );
   virtual ~DBIHandleFirebird();

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

   isc_db_handle& getConn() { return m_conn; }
   const isc_db_handle& getConn() const { return m_conn; }

   isc_tr_handle& getTr() { return m_tr; }
   const isc_tr_handle& getTr() const { return m_tr; }

   // Throws a DBI error, using the last error code and description.
   void throwError( const char* file, int line, int code );
};


class DBIStatementFirebird : public DBIStatement
{
protected:
   isc_stmt_handle m_statement;
   FirebirdDBIInBind* m_inBind;

public:
   DBIStatementFirebird( DBIHandle *dbh, const isc_stmt_handle& stmt );
   virtual ~DBIStatementFirebird();

   virtual int64 execute( const ItemArray& params );
   virtual void reset();
   virtual void close();

   DBIHandleFirebird* getMySql() const { return static_cast<DBIHandleFirebird*>( m_dbh ); }
   const isc_stmt_handle& my_statement() const { return m_statement; }
   isc_stmt_handle& my_statement() { return m_statement; }
};


class DBIServiceFirebird : public DBIService
{
public:
   DBIServiceFirebird() : DBIService( "DBI_fbsql" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceFirebird theFirebirdService;

#endif /* FALCON_FIREBIRD_H */

/* end of fbsql_mod.h */

