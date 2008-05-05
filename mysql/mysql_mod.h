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

#include "../include/dbiservice.h"

#include <mysql.h>

namespace Falcon
{

class DBIRecordsetMySQL : public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   MYSQL_RES *m_res;
   MYSQL_FIELD *m_fields;
   MYSQL_ROW m_rowData;
   unsigned long *m_fieldLengths;

   static dbi_type getFalconType( int typ );

public:
   DBIRecordsetMySQL( DBIHandle *dbh, MYSQL_RES *res );
   ~DBIRecordsetMySQL();

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


class DBITransactionMySQL : public DBITransaction
{
protected:
   bool m_inTransaction;

public:
   DBITransactionMySQL( DBIHandle *dbh );

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

class DBIHandleMySQL : public DBIHandle
{
protected:
   MYSQL *m_conn;

   DBITransactionMySQL *m_connTr;

public:
   DBIHandleMySQL();
   DBIHandleMySQL( MYSQL *conn );
   virtual ~DBIHandleMySQL() {}

   MYSQL *getConn() { return m_conn; }

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

class DBIServiceMySQL : public DBIService
{
public:
   DBIServiceMySQL() : DBIService( "DBI_mysql" ) {}

   virtual dbi_status init();
   virtual DBIHandle *connect( const String &parameters, bool persistent,
                               dbi_status &retval, String &errorMessage );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServiceMySQL theMySQLService;

#endif /* DBI_MYSQL_H */

/* end of mysql.h */

