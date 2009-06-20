/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3.h
 *
 * SQLite3 driver main module interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_SQLITE3_MOD_H
#define DBI_SQLITE3_MOD_H

#include <sqlite3.h>

#include "../include/dbiservice.h"

namespace Falcon
{

class DBIRecordsetSQLite3 : public DBIRecordset
{
protected:
   int m_row;
   int m_columnCount;

   sqlite3_stmt *m_res;
   bool m_bHasRow;

   static dbi_type getFalconType( int typ );

public:
   DBIRecordsetSQLite3( DBIHandle *dbh, sqlite3_stmt *res, bool bHasRow = false );
   ~DBIRecordsetSQLite3();

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
};

class DBITransactionSQLite3 : public DBITransaction
{
protected:
   bool m_inTransaction;

public:
   DBITransactionSQLite3( DBIHandle *dbh );

   virtual DBIRecordset *query( const String &query, int64 &affected, dbi_status &retval );
   virtual dbi_status begin();
   virtual dbi_status commit();
   virtual dbi_status rollback();
   virtual dbi_status close();
   virtual dbi_status getLastError( String &description );
   virtual DBIBlobStream *openBlob( const String &blobId, dbi_status &status );
   virtual DBIBlobStream *createBlob( dbi_status &status, const String &params= "",
      bool bBinary = false );
};

class DBIHandleSQLite3 : public DBIHandle
{
protected:
   sqlite3 *m_conn;

   DBITransactionSQLite3 *m_connTr;

public:
   DBIHandleSQLite3();
   DBIHandleSQLite3( sqlite3 *conn );
   virtual ~DBIHandleSQLite3();

   sqlite3 *getConn() { return m_conn; }

   virtual DBITransaction *startTransaction();
   virtual DBITransaction *getDefaultTransaction();
   virtual dbi_status closeTransaction( DBITransaction *tr );
   virtual int64 getLastInsertedId();
   virtual int64 getLastInsertedId( const String &value );

   virtual dbi_status escapeString( const String &value, String &escaped );
   virtual dbi_status close();
};

class DBIServiceSQLite3 : public DBIService
{
public:
   DBIServiceSQLite3() : DBIService( "DBI_sqlite3" ) {}

   virtual dbi_status init();
   virtual DBIHandle *connect( const String &parameters, bool persistent,
                               dbi_status &retval, String &errorMessage );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceSQLite3 theSQLite3Service;

#endif /* DBI_SQLITE3_H */

/* end of sqlite3.h */

