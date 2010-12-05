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
#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

namespace Falcon
{

class DBIHandleSQLite3;

class SQLite3StatementHandler {
public:
   
   SQLite3StatementHandler( sqlite3_stmt* hStmt ):
       m_hStmt(hStmt),
       m_nRefCount(1)
   {}
   
   ~SQLite3StatementHandler() {
      sqlite3_finalize( m_hStmt );
   }

   void incref() { m_nRefCount ++; }

   void decref() { if ( --m_nRefCount == 0 ) delete this; }

   sqlite3_stmt* handle() const { return m_hStmt; }

private:
   sqlite3_stmt* m_hStmt;
   int m_nRefCount;
};


class Sqlite3InBind: public DBIInBind
{

public:
   Sqlite3InBind(sqlite3_stmt* stmt);
   virtual ~Sqlite3InBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

private:
   sqlite3_stmt* m_stmt;
};


class DBIRecordsetSQLite3 : public DBIRecordset
{
protected:
   int m_row;
   int m_columnCount;
   SQLite3StatementHandler *m_pStmt;
   sqlite3_stmt* m_stmt;
   bool m_bAsString;

public:
   DBIRecordsetSQLite3( DBIHandle *dbt, SQLite3StatementHandler* pStmt );
   DBIRecordsetSQLite3( DBIHandle *dbt, sqlite3_stmt* stmt );
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
   SQLite3StatementHandler* m_pStmt;
   Sqlite3InBind m_inBind;

public:
   DBIStatementSQLite3( DBIHandleSQLite3 *dbh, SQLite3StatementHandler* pStmt );
   DBIStatementSQLite3( DBIHandleSQLite3 *dbh, sqlite3_stmt* stmt );
   virtual ~DBIStatementSQLite3();

   virtual DBIRecordset* execute( ItemArray* params );
   virtual void reset();
   virtual void close();

   sqlite3_stmt* sqlite3_statement() const { return m_statement; }
};


class DBIHandleSQLite3 : public DBIHandle
{
protected:
   sqlite3 *m_conn;
   DBISettingParams m_settings;
   bool m_bInTrans;

   sqlite3_stmt* int_prepare( const String &query ) const;
   void int_execute( sqlite3_stmt* pStmt, ItemArray* params );

public:
   DBIHandleSQLite3();
   DBIHandleSQLite3( sqlite3 *conn );
   virtual ~DBIHandleSQLite3();

   virtual void options( const String& params );
   virtual const DBISettingParams* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, ItemArray* params=0 );
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

class DBIServiceSQLite3 : public DBIService
{
public:
   DBIServiceSQLite3();

   virtual void init();
   virtual DBIHandle *connect( const String &parameters );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceSQLite3 theSQLite3Service;

#endif

/* end of sqlite3.h */
