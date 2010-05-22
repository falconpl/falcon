/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql.h
 *
 * MySQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
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

class DBIBindMySQL: public DBIBind
{

public:
   DBIBindMySQL();

   virtual ~DBIBindMySQL();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   MYSQL_BIND* mybindings() const { return m_mybind; }

private:
   MYSQL_BIND* m_mybind;
};


class DBIRecordsetMySQL : public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   MYSQL_RES *m_res;
   MYSQL_STMT *m_stmt;
   MYSQL_FIELD* m_fields;

   //static dbi_type getFalconType( int typ );

   // Binding data
   MYSQL_BIND* m_pMyBind;
   DBIOutBind* m_pOutBind;

   // used to keep track of blobs that must be zeroed before fetch
   int* m_pBlobId;
   int m_nBlobCount;
public:
   DBIRecordsetMySQL( DBITransaction *dbt, MYSQL_RES *res, MYSQL_STMT *stmt );
   ~DBIRecordsetMySQL();

   virtual bool fetchRow();
   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();

};


class DBIHandleMySQL : public DBIHandle
{
protected:
   MYSQL *m_conn;
   DBISettingParams m_settings;

public:
   DBIHandleMySQL();
   DBIHandleMySQL( MYSQL *conn );
   virtual ~DBIHandleMySQL();

   virtual bool setTransOpt( const String& params );
   virtual const DBISettingParams* transOpt() const;

   MYSQL *getConn() { return m_conn; }

   virtual DBITransaction* startTransaction( const String& options );
   virtual void close();

   // Throws a DBI error, using the last error code and description.
   void throwError( const char* file, int line, int code );
};


class DBITransactionMySQL : public DBITransaction
{
protected:
   bool m_inTransaction;
   MYSQL_STMT* m_statement;
   DBIBindMySQL* m_inBind;

public:
   DBITransactionMySQL( DBIHandle *dbh, DBISettingParams* settings );
   virtual ~DBITransactionMySQL();

   virtual DBIRecordset *query( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual void call( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual void prepare( const String &query );
   virtual void execute( const ItemArray& params, int64 &affectedRows );

   virtual DBITransaction* startTransaction( const String& settings );
   virtual void begin();
   virtual void commit();
   virtual void rollback();
   virtual void close();
   virtual int64 getLastInsertedId( const String& name = "" );

   DBIHandleMySQL* getMySql() const { return static_cast<DBIHandleMySQL*>( m_dbh ); }
};


class DBIServiceMySQL : public DBIService
{
public:
   DBIServiceMySQL() : DBIService( "DBI_mysql" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters, bool persistent );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceMySQL theMySQLService;

#endif /* DBI_MYSQL_H */

/* end of mysql.h */

