/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_mod.h

   SQLite3 driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:23:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_DBI_SQLITE3_MOD_H
#define FALCON_DBI_SQLITE3_MOD_H

#include <sqlite3.h>

namespace Falcon
{

class Sqlite3InBind: public DBIInBind
{

public:
   Sqlite3InBind();
   virtual ~Sqlite3InBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

private:
};


class DBIRecordsetSQLite3 : public DBIRecordset
{
protected:
   int m_row;
   int m_columnCount;

   sqlite3_stmt *m_res;
   bool m_bHasRow;

public:
   DBIRecordsetSQLite3( DBIStatement *dbt, sqlite3_stmt* stmt );
   virtual ~DBIRecordsetSQLite3();

   virtual bool fetchRow();
   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();
};

class DBITransactionSQLite3 : public DBIStatement
{
protected:
   bool m_inTransaction;

public:
   DBITransactionSQLite3( DBIHandle *dbh );

   virtual ~DBITransactionSQLite3();

   virtual DBIRecordset *query( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual void call( const String &sql, int64 &affectedRows, const ItemArray& params );
   virtual void prepare( const String &query );
   virtual void execute( const ItemArray& params, int64 &affectedRows );

   virtual DBIStatement* startTransaction( const String& settings );
   virtual void begin();
   virtual void commit();
   virtual void rollback();
   virtual void close();
   virtual int64 getLastInsertedId( const String& name = "" );

   DBIHandleSQLite3* getSQLite() const { return static_cast<DBIHandleSQLite3*>( m_dbh ); }
};

class DBIHandleSQLite3 : public DBIHandle
{
protected:
   sqlite3 *m_conn;

public:
   DBIHandleSQLite3();
   DBIHandleSQLite3( sqlite3 *conn );
   virtual ~DBIHandleSQLite3();

   virtual bool setTransOpt( const String& params );
   virtual const DBISettingParams* transOpt() const;
   virtual DBIStatement* startTransaction( const String& options );
   virtual void close();

   sqlite3 *getConn() { return m_conn; }
};

class DBIServiceSQLite3 : public DBIService
{
public:
   DBIServiceSQLite3() : DBIService( "DBI_sqlite3" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters, bool persistent );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

extern DBIServiceSQLite3 theSQLite3Service;

}


#endif

/* end of sqlite3.h */
