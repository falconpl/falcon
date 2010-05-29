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

namespace Falcon
{

class MyDBIInBind: public DBIInBind
{

public:
   MyDBIInBind();

   virtual ~MyDBIInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   MYSQL_BIND* mybindings() const { return m_mybind; }

private:
   MYSQL_BIND* m_mybind;
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

   // Binding data
   MYSQL_BIND* m_pMyBind;
   MyDBIOutBind* m_pOutBind;

   // used to keep track of blobs that must be zeroed before fetch
   int* m_pBlobId;
   int m_nBlobCount;

public:
   DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbt, MYSQL_RES *res, MYSQL_STMT *stmt, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_STMT();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();
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

   virtual bool getColumnValue( int nCol, Item& value );
};


class DBIHandleMySQL : public DBIHandle
{
protected:
   MYSQL *m_conn;
   DBISettingParams m_settings;

   MYSQL_STMT* my_prepare( const String &query );
   int64 my_execute( MYSQL_STMT* stmt, MyDBIInBind& bindings, const ItemArray& params );

public:
   DBIHandleMySQL();
   DBIHandleMySQL( MYSQL *conn );
   virtual ~DBIHandleMySQL();

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

   MYSQL *getConn() { return m_conn; }

   // Throws a DBI error, using the last error code and description.
   void throwError( const char* file, int line, int code );
};


class DBIStatementMySQL : public DBIStatement
{
protected:
   MYSQL_STMT* m_statement;
   MyDBIInBind* m_inBind;

public:
   DBIStatementMySQL( DBIHandle *dbh, MYSQL_STMT* stmt );
   virtual ~DBIStatementMySQL();

   virtual int64 execute( const ItemArray& params );
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

/* end of mysql.h */

