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

#ifndef PGSQL_MOD_H
#define PGSQL_MOD_H

#include <libpq-fe.h>

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

namespace Falcon
{

#if 0
class PgSQLInBind
    :
    public DBIInBind
{
public:

    PgSQLInBind();
    virtual ~PgSQLInBind();

    virtual void onFirstBinding( int size );
    virtual void onItemChanged( int num );

};
#endif

class DBIHandlePgSQL;


void dbi_pgsqlQuestionMarksToDollars( const String& input, String& output );


class DBIRecordsetPgSQL
    :
    public DBIRecordset
{
protected:

    int64       m_row;
    int64       m_rowCount;
    int         m_columnCount;

    PGresult*   m_res;

public:

    DBIRecordsetPgSQL( DBIHandlePgSQL* dbh, PGresult* res );
    virtual ~DBIRecordsetPgSQL();

    virtual bool fetchRow();
    virtual int64 getRowIndex();
    virtual int64 getRowCount();
    virtual int getColumnCount();
    virtual bool getColumnName( int nCol, String& name );
    virtual bool getColumnValue( int nCol, Item& value );
    virtual bool discard( int64 ncount );
    virtual void close();

};


class DBIStatementPgSQL
    :
    public DBIStatement
{
protected:

    int32   m_nParams;
    String  m_execString;
    AutoCString m_zExecString;

    void getExecString( int32 nParams );

public:

    DBIStatementPgSQL( DBIHandlePgSQL* dbh, const String& query );
    virtual ~DBIStatementPgSQL();

    virtual int64 execute( const ItemArray& params );
    virtual void reset();
    virtual void close();
};


#if 0
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
#endif


class DBIHandlePgSQL
    :
    public DBIHandle
{
protected:

    PGconn* m_conn;
    bool    m_bInTrans;
    DBISettingParams m_settings;

public:

    DBIHandlePgSQL( PGconn* = 0 );
    virtual ~DBIHandlePgSQL();

    PGconn* getConn() { return m_conn; }

    virtual void options( const String& params );
    virtual const DBISettingParams* options() const;

    virtual void begin();
    virtual void commit();
    virtual void rollback();

    virtual DBIRecordset* query( const String &sql, int64 &affectedRows, const ItemArray& params );
    virtual void perform( const String &sql, int64 &affectedRows, const ItemArray& params );
    virtual DBIRecordset* call( const String &sql, int64 &affectedRows, const ItemArray& params );
    virtual DBIStatement* prepare( const String &query );
    virtual int64 getLastInsertedId( const String& name = "" );

    virtual void selectLimited( const String& query, int64 nBegin, int64 nCount, String& result );

    virtual void close();

    static void throwError( const char* file, int line, PGresult* res );
    PGresult* internal_exec( const String& sql, int64& affectedRows );
};

#if 0
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
   virtual void close();
   virtual DBIStatement* getDefaultTransaction();
};
#endif


class DBIServicePgSQL
    :
    public DBIService
{
public:

    DBIServicePgSQL()
        :
        DBIService( "DBI_pgsql" )
    {}

    virtual void init();

    virtual DBIHandle* connect( const String& parameters );

    virtual CoreObject* makeInstance( VMachine* vm, DBIHandle* dbh );

};

}

extern Falcon::DBIServicePgSQL thePgSQLService;

#endif /* PGSQL_MOD_H */
