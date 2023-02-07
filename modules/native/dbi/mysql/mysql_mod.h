/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_mod.h
 *
 * MySQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 16:58:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_MYSQL_H
#define DBI_MYSQL_H

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

#include <mysql.h>

#ifndef IS_LONGDATA
#define IS_LONGDATA(t) ((t) >= MYSQL_TYPE_TINY_BLOB && (t) <= MYSQL_TYPE_STRING)
#endif

namespace Falcon
{

class MYSQLHandle: public DBIRefCounter<MYSQL*> {
public:
   MYSQLHandle( MYSQL* m ):
      DBIRefCounter<MYSQL*>( m )
   {}

   virtual ~MYSQLHandle()
   {
      mysql_close( handle() );
   }
};


class MYSQLStmtHandle: public DBIRefCounter<MYSQL_STMT*> {
public:
   MYSQLStmtHandle( MYSQL_STMT* m ):
      DBIRefCounter<MYSQL_STMT*>( m )
   {}

   virtual ~MYSQLStmtHandle();
};

class MyDBIInBind: public DBIInBind
{

public:
   MyDBIInBind( MYSQL_STMT* stmt );

   virtual ~MyDBIInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   MYSQL_BIND* mybindings() const { return m_mybind; }

private:
   MYSQL_BIND* m_mybind;
   MYSQL_STMT* m_stmt;
};


class MyDBIOutBind: public DBIOutBind
{
public:
   MyDBIOutBind():
      bIsNull( false ),
      nLength( 0 )
   {}

   ~MyDBIOutBind() {}

   my_bool bIsNull;
   unsigned long nLength;
};

class DBIHandleMySQL;

class DBIRecordsetMySQL: public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   MYSQL_RES *m_res;
   MYSQL_FIELD* m_fields;

   bool m_bCanSeek;

   MYSQLHandle *m_pConn;

public:
   DBIRecordsetMySQL( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual void close();
};

class DBIRecordsetMySQL_STMT: public DBIRecordsetMySQL
{
protected:
   MYSQL_STMT *m_stmt;
   MYSQLStmtHandle *m_pStmt;

   // Binding data
   MYSQL_BIND* m_pMyBind;
   MyDBIOutBind* m_pOutBind;

   // used to keep track of blobs that must be zeroed before fetch
   int* m_pBlobId;
   int m_nBlobCount;

public:
   DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbt, MYSQL_RES *res, MYSQLStmtHandle *pStmt, bool bCanSeek = false );
   DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbt, MYSQL_RES *res, MYSQL_STMT *stmt, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_STMT();

   void init();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();

   /** This cind of recordsets can generate a next recordset. */
   virtual DBIRecordset* getNext();
};


class DBIRecordsetMySQL_RES : public DBIRecordsetMySQL
{
protected:
   MYSQL_ROW m_rowData;
   CoreObject* makeTimestamp( const String& str );

public:
   DBIRecordsetMySQL_RES( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_RES();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
};

class DBIRecordsetMySQL_RES_STR: public DBIRecordsetMySQL_RES
{
public:
   DBIRecordsetMySQL_RES_STR( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_RES_STR();

   virtual void close();

   virtual bool getColumnValue( int nCol, Item& value );
};


class DBIHandleMySQL : public DBIHandle
{
protected:
   MYSQL *m_conn;
   MYSQLHandle *m_pConn;
   DBISettingParams m_settings;

   MYSQL_STMT* my_prepare( const String &query, bool bCanFallback = false );
   int64 my_execute( MYSQL_STMT* stmt, MyDBIInBind& bindings, ItemArray* params );

   DBIRecordset *query_internal( const String &sql, ItemArray* params=0 );

public:
   DBIHandleMySQL();
   DBIHandleMySQL( MYSQL *conn );
   virtual ~DBIHandleMySQL();

   virtual void options( const String& params );
   virtual const DBISettingParams* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, ItemArray* params=0 );
   virtual void result( const String &sql, Item& res, ItemArray* params =0 );

   virtual DBIStatement* prepare( const String &query );
   virtual int64 getLastInsertedId( const String& name = "" );

   virtual void begin();
   virtual void commit();
   virtual void rollback();

   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result );

   MYSQLHandle *getConn() { return m_pConn; }

   // Throws a DBI error, using the last error code and description.
   void throwError( const char* file, int line, int code );
};


class DBIStatementMySQL : public DBIStatement
{
protected:
   MYSQL_STMT* m_statement;
   MYSQLHandle* m_pConn;
   MYSQLStmtHandle *m_pStmt;
   MyDBIInBind* m_inBind;
   bool m_bBound;

public:
   DBIStatementMySQL( DBIHandleMySQL *dbh, MYSQL_STMT* stmt );
   virtual ~DBIStatementMySQL();

   virtual DBIRecordset* execute( ItemArray* params );
   virtual void reset();
   virtual void close();

   DBIHandleMySQL* getMySql() const { return static_cast<DBIHandleMySQL*>( m_dbh ); }
   MYSQL_STMT* my_statement() const { return m_statement; }
};


class DBIServiceMySQL : public DBIService
{
public:
   DBIServiceMySQL() : DBIService( "DBI_mysql" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceMySQL theMySQLService;

#endif /* DBI_MYSQL_H */

/* end of mysql_mod.h */

