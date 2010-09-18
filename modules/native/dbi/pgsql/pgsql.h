/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql.h
 *
 * Pgsql driver main module interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Sun Dec 23 21:36:20 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_PGSQL_H
#define DBI_PGSQL_H

#include <libpq-fe.h>

#include "../include/dbiservice.h"

namespace Falcon
{

class DBIRecordsetPgSQL : public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   PGresult *m_res;

   static dbi_type getFalconType( Oid pgType );

public:
   DBIRecordsetPgSQL( DBIHandle *dbh, PGresult *res );
   ~DBIRecordsetPgSQL();

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

class DBITransactionPgSQL : public DBIStatement
{
protected:
   bool m_inTransaction;

public:
   DBITransactionPgSQL( DBIHandle *dbh );

   virtual DBIRecordset *query( const String &query, int64 &affected_rows, dbi_status &retval );
   virtual dbi_status begin();
   virtual dbi_status commit();
   virtual dbi_status rollback();
   virtual void close();
   virtual dbi_status getLastError( String &description );
   virtual DBIBlobStream *openBlob( const String &blobId, dbi_status &status );
   virtual DBIBlobStream *createBlob( dbi_status &status, const String &params= "",
      bool bBinary = false );
};

class DBIHandlePgSQL : public DBIHandle
{
protected:
   PGconn *m_conn;

   DBITransactionPgSQL *m_connTr;

public:
   DBIHandlePgSQL();
   DBIHandlePgSQL( PGconn *conn );
   virtual ~DBIHandlePgSQL();

   PGconn *getPGconn() { return m_conn; }

   virtual DBIStatement *startTransaction();
   virtual dbi_status closeTransaction( DBIStatement *tr );
   virtual int64 getLastInsertedId();
   virtual int64 getLastInsertedId( const String &value );
   virtual dbi_status getLastError( String &description );
   virtual dbi_status escapeString( const String &value, String &escaped );
   virtual dbi_status close();
   virtual DBIStatement* getDefaultTransaction();
};

class DBIServicePgSQL : public DBIService
{
public:
   DBIServicePgSQL() : DBIService( "DBI_pgsql" ) {}

   virtual dbi_status init();
   virtual DBIHandle *connect( const String &parameters, bool persistent,
                               dbi_status &retval, String &errorMessage );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServicePgSQL thePgSQLService;

#endif /* DBI_PGSQL_H */

/* end of pgsql.h */

