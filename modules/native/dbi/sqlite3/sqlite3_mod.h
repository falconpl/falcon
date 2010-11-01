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


class DBIHandleSQLite3 : public DBIHandle
{
protected:
   sqlite3 *m_conn;
   DBISettingParams m_settings;
   bool m_bInTrans;

   sqlite3_stmt* int_prepare( const String &query ) const;
   void int_execute( sqlite3_stmt* pStmt, const ItemArray& params );

public:
   DBIHandleSQLite3();
   DBIHandleSQLite3( sqlite3 *conn );
   virtual ~DBIHandleSQLite3();

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
