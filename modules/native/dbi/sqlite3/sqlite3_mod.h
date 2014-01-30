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
#include <falcon/dbi_service.h>

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
   // caching for simpler access
   sqlite3_stmt* m_stmt;
   bool m_bAsString;

public:
   DBIRecordsetSQLite3( DBIHandleSQLite3 *dbt, sqlite3_stmt* stmt );
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
public:
   DBIStatementSQLite3( DBIHandleSQLite3 *dbh, sqlite3_stmt* stmt );
   virtual ~DBIStatementSQLite3();

   virtual DBIRecordset* execute( ItemArray* params );
   virtual void reset();
   virtual void close();

   sqlite3_stmt* sqlite3_statement() const { return m_statement; }

protected:
   sqlite3_stmt* m_statement;
   Sqlite3InBind m_inBind;
   bool m_bFirst;
};


class DBIHandleSQLite3 : public DBIHandle
{
protected:
   sqlite3* m_conn;
   DBISettingParams m_settings;
   bool m_bInTrans;

   sqlite3_stmt* int_prepare( const String &query ) const;

public:
   DBIHandleSQLite3( const Class* h );
   DBIHandleSQLite3( const Class* h, sqlite3 *conn );
   virtual ~DBIHandleSQLite3();

   virtual void connect( const String& params );
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
};


class DBIServiceSQLite3 : public DBIService
{
public:
   DBIServiceSQLite3(Module* mod);
   virtual DBIHandle *connect( const String &parameters );
};

}

#endif

/* end of sqlite3.h */
